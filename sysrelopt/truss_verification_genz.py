#type:ignore
"""Verification - Genz's method"""
import numpy as np, time
import matplotlib.pyplot as plt
from scipy.stats import norm, qmc
from scipy.optimize import minimize

from truss_rbdo_bounds import comps_e, comps_p, pf_bounds, opt as opt_base
from truss_rbdo_elastic_system import SQ2, QP, elastic_q, pf_relaxed, opt_rel

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 12})

# Author: James Whiteley (github.com/jamesalexwhiteley)

BT = 3.0              # reliability index
K_CHK, R_CHK = 18, 8  # sobol 2^k, x 
K_OPT, R_OPT = 15, 2  # cheaper, inside re-optimisation
CY = CF = .1

# ---------- Genz multinormal ----------
_PTS = {}
def sobol_pts(d, k, seed):
    key = (d, k, seed)
    if key not in _PTS:
        _PTS[key] = qmc.Sobol(d, scramble=True, seed=seed).random_base2(k)
    return _PTS[key]

def _genz_once(b, C, w):
    """One Genz sweep: pf = 1 - Phi_m(b; rho)."""
    m, N = b.size, len(w)
    y = np.empty((m - 1, N))
    z = b[0] / C[0, 0]
    e = np.full(N, norm.cdf(z))
    tot = np.full(N, norm.cdf(-z))  # i = 0 term
    P = np.ones(N)  # prod_{j<i} e_j
    for i in range(1, m):
        y[i-1] = norm.ppf(np.clip(w[:, i-1] * e, 1e-16, 1 - 1e-16))
        P = P * e
        z = (b[i] - C[i, :i] @ y[:i]) / C[i, i]
        e = norm.cdf(z)
        tot += norm.cdf(-z) * P
    return tot.mean()

def genz_pf(b, rho, k=K_CHK, reps=R_CHK):
    """(pf, standard error of pf) with pf = 1 - Phi_m(b; rho)."""
    b = np.atleast_1d(np.asarray(b, float)); m = b.size
    if m == 1:
        return float(norm.cdf(-b[0])), 0.
    o = np.argsort(b)  # most critical first
    b = b[o]; rho = np.asarray(rho, float)[np.ix_(o, o)]
    C = np.linalg.cholesky(rho + 1e-12*np.eye(m))  # rho can be rank m-1
    v = np.array([_genz_once(b, C, sobol_pts(m-1, k, s)) for s in range(reps)])
    return float(v.mean()), float(v.std(ddof=1)/np.sqrt(reps)) if reps > 1 else 0.

def pf2beta(pf, bmin=np.inf):
    return min(-norm.ppf(np.clip(pf, 1e-300, 1 - 1e-16)), bmin)

# ---------- component sets -> (beta, rho) ----------
def series_bt(idx, t, cY=CY, cF=CF):
    """Margins M_i = Y_i - t_i F  (i.e. R_i Y_i - q_i F scaled by 1/R_i)."""
    t = np.asarray(t, float); s = np.hypot(cY, t*cF)
    al = np.zeros((t.size, 4))
    al[np.arange(t.size), np.asarray(idx, int)] = cY / s
    al[:, 3] = -t * cF / s
    rho = al @ al.T; np.fill_diagonal(rho, 1.)
    return (1 - t) / s, rho

def relaxed_thresholds(a1, a2, gamma):
    """(idx, t) for the elastic set A, the snapped set B and union A_i U B_i."""
    B = ([1, 2], QP / np.array([a2, a1]))  # snapped forces, bars 2,3
    if gamma >= 1 - 1e-12:
        return None, B, None  # bar 1 gone: series over B
    Rb = np.array([(1-gamma)*a1, a2, a1])
    tA = elastic_q(a1, a2, gamma) / Rb
    U = ([0, 1, 2], np.concatenate([tA[:1], np.maximum(tA[1:], B[1])]))
    return ([0, 1, 2], tA), B, U

# ---------- Genz evaluators ----------
def genz_elem(x, gamma, comps, **kw):
    """(pf, se, min beta_i) of the element-level series system."""
    b, rho = comps(np.array([x[0], x[1], x[0]]), np.array([gamma, 0., 0.]))
    pf, se = genz_pf(b, rho, **kw)
    return pf, se, b.min()

def genz_relaxed(x, gamma, **kw):
    """(pf, se) of the elastic system-level (snap-allowed) event."""
    A, B, U = relaxed_thresholds(x[0], x[1], gamma)
    pfB, seB = genz_pf(*series_bt(*B), **kw)
    if A is None:
        return pfB, seB
    pfA, seA = genz_pf(*series_bt(*A), **kw)
    pfU, seU = genz_pf(*series_bt(*U), **kw)
    return pfA + pfB - pfU, float(np.hypot(np.hypot(seA, seB), seU))

def beta_elem_genz(x, gamma, comps, **kw):
    pf, _, bmin = genz_elem(x, gamma, comps, **kw)
    return pf2beta(pf, bmin)

def beta_rel_genz(x, gamma, **kw):
    return pf2beta(genz_relaxed(x, gamma, **kw)[0])

# ---------- RBDO re-solved with the Genz evaluator ----------
def opt_genz(kind, gamma, x0s, bt=BT):
    """Pf from Genz instead."""
    V = lambda x: 2*SQ2*x[0] + x[1]
    kw = dict(k=K_OPT, reps=R_OPT)
    if kind == 'relaxed':  # undamaged constraint is the
        g0 = lambda x: beta_elem_genz(x, 0., comps_e, **kw)  # element-level one
        g1 = lambda x, g=gamma: beta_rel_genz(x, g, **kw)
    else:
        c = comps_e if kind == 'elastic' else comps_p
        g0 = lambda x, c=c: beta_elem_genz(x, 0., c, **kw)
        g1 = lambda x, g=gamma, c=c: beta_elem_genz(x, g, c, **kw)
    cons = [{'type': 'ineq', 'fun': lambda x: g0(x) - bt},
            {'type': 'ineq', 'fun': lambda x: g1(x) - bt}]
    best = None
    for x0 in x0s:
        r = minimize(V, x0, method='SLSQP', bounds=[(1e-3, 6.)]*2,
                     constraints=cons, options={'ftol': 1e-12, 'maxiter': 200})
        if min(g0(r.x), g1(r.x)) - bt > -1e-4 and (best is None or r.fun < best.fun):
            best = r
    return best

# ---------- checks ----------
def check_point(kind, gamma, x):
    """Analytic vs Genz at one design point."""
    if kind == 'relaxed':  # Gauss-Hermite 
        pf_an = float(pf_relaxed(x[0], x[1], gamma))
        lo = hi = np.nan
        pf_gz, se = genz_relaxed(x, gamma)
        b_gz = pf2beta(pf_gz)
        ok = abs(pf_gz - pf_an) <= max(2*se, 1e-6*pf_an)
    else:  # Ditlevsen 
        comps = comps_e if kind == 'elastic' else comps_p
        b, rho = comps(np.array([x[0], x[1], x[0]]), np.array([gamma, 0., 0.]))
        lo, hi = pf_bounds(b, rho)
        pf_an = hi
        pf_gz, se, bmin = genz_elem(x, gamma, comps)
        b_gz = pf2beta(pf_gz, bmin)
        ok = lo - 2*se <= pf_gz <= hi + 2*se
    return dict(pf_an=pf_an, pf_gz=pf_gz, se=se, b_gz=b_gz, lo=lo, hi=hi, ok=ok)

if __name__ == '__main__':
    t0 = time.time()

    def opt_analytic(kind, gamma, x0s):
        if kind == 'relaxed':
            return opt_rel(gamma, x0s)
        return opt_base(comps_e if kind == 'elastic' else comps_p, gamma, x0s=x0s)

    # ---- analytic curves ----
    gs = np.concatenate([np.linspace(0, .95, 20), np.linspace(.96, .9999, 6)])
    curves, ends = {}, {}
    for kind in ('elastic', 'relaxed', 'plastic'):
        xp = [3., 3.]; v = []
        for g in gs:
            r = opt_analytic(kind, g, [xp, [1., 1.], [5., 5.]]); xp = r.x
            v.append(r.fun)
        curves[kind] = np.array(v)
        ends[kind] = opt_analytic(kind, 1., [[2.5, 1.5], [1., 1.], [5., 5.]])
    print(f"analytic sweeps: {time.time()-t0:.1f} s")

    # ---- verification at a few damage levels ----
    gchk = np.array([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.])
    HB = (f"{'curve':<9}{'gamma':>6}{'a1*':>8}{'a2*':>8}{'V*':>8}"
          f"{'pf_lo':>13}{'pf_hi':>13}{'pf_genz':>13}{'2se':>9}{'in?':>5}"
          f"{'b_genz':>8}{'V*_genz':>9}{'dV%':>7}")
    HR = (f"{'curve':<9}{'gamma':>6}{'a1*':>8}{'a2*':>8}{'V*':>8}"
          f"{'pf_gaussherm':>13}{'pf_genz':>13}{'2se':>9}{'rel.err':>9}{'ok?':>5}"
          f"{'b_genz':>8}{'V*_genz':>9}{'dV%':>7}")
    marks, npass, nchk = {}, 0, 0
    for kind in ('elastic', 'relaxed', 'plastic'):
        hdr = HR if kind == 'relaxed' else HB
        print('\n' + hdr); print('-'*len(hdr))
        vg = []; xp = [3., 3.]
        for g in gchk:
            r = ends[kind] if g == 1. else opt_analytic(kind, g, [xp, [1., 1.]])
            xa = r.x
            if g < .99: xp = xa
            c = check_point(kind, g, xa)
            nchk += 1; npass += c['ok']
            rg = opt_genz(kind, g, [xa, 1.05*np.asarray(xa)])
            Va, Vg = r.fun, (rg.fun if rg is not None else np.nan)
            vg.append(Vg)
            row = (f"{kind:<9}{g:>6.2f}{xa[0]:>8.4f}{xa[1]:>8.4f}{Va:>8.4f}")
            if kind == 'relaxed':
                row += (f"{c['pf_an']:>13.6e}{c['pf_gz']:>13.6e}{2*c['se']:>9.1e}"
                        f"{abs(c['pf_gz']/c['pf_an']-1):>9.1e}")
            else:
                row += (f"{c['lo']:>13.6e}{c['pf_an']:>13.6e}"
                        f"{c['pf_gz']:>13.6e}{2*c['se']:>9.1e}")
            print(row + f"{'y' if c['ok'] else 'N':>5}{c['b_gz']:>8.4f}"
                        f"{Vg:>9.4f}{100*(Vg-Va)/Va:>7.2f}")
        marks[kind] = np.array(vg)
    print(f"total: {time.time()-t0:.1f} s")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(gs, curves['elastic'], '-.', color='tab:blue', 
            label=r'elastic (element-level), $\gamma<1$')
    ax.plot([gs[-1], 1], [curves['elastic'][-1], ends['elastic'].fun], '--',
            color='tab:blue', lw=.5)
    ax.plot(gs, curves['relaxed'], ':', color='tab:blue', lw=2,
            label='elastic (system-level)')
    ax.plot([gs[-1], 1], [curves['relaxed'][-1], ends['relaxed'].fun], ':',
            color='tab:blue', lw=2)
    ax.plot(gs, curves['plastic'], 'k-', label='plastic (system-level)')
    ax.plot([gs[-1], 1], [curves['plastic'][-1], ends['plastic'].fun], 'k-')
    lab = r'Genz ($%d \times 2^{%d}$ points)' % (R_OPT, K_OPT)
    for kind, mk in [('elastic', 's'), ('relaxed', 's'), ('plastic', 's')]:
        ax.plot(gchk, marks[kind], mk, mfc='none', mec='0.3', ms=6, ls='none',
                label=lab if kind == 'elastic' else None)
    ax.set_xlabel(r'Damage $\gamma$')
    ax.set_ylabel(r'Optimum volume $V^*/V^*_{\max}$')
    ax.legend(); ax.grid(alpha=.3); plt.tight_layout()
    plt.savefig('pfvol_gamma_verify_genz.png', dpi=400)
    plt.show()
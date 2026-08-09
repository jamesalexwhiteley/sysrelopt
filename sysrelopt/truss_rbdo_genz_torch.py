#type:ignore
"""Gradient-based RBDO of a rigid-plastic model (via Genz's algorithm)."""
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch
from torch.special import ndtr, ndtri
import time

torch.set_default_dtype(torch.float64)  # tail probabilities need float64

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 12})

# Author: James Whiteley (github.com/jamesalexwhiteley)

# ---------- multinormal survival pf = 1 - Phi_m(b; rho) ----------
_PTS = {}
def sobol_pts(d, n_log2):
    key = (d, n_log2)
    if key not in _PTS:
        _PTS[key] = torch.tensor(qmc.Sobol(d, scramble=False).random_base2(n_log2))
    return _PTS[key]

def genz_sf(b, rho, n_log2=16):
    """pf = 1 - Phi_m(b; rho)."""
    m = len(b)
    C = torch.linalg.cholesky(rho)
    w = sobol_pts(m - 1, n_log2); N = len(w)
    z = b[0] / C[0, 0]
    e = [ndtr(z).expand(N)]; s = [ndtr(-z).expand(N)]
    y = []
    for i in range(1, m):
        u = torch.clamp(w[:, i - 1] * e[i - 1], 1e-15, 1 - 1e-15)
        y.append(ndtri(u))  # clamp's zero-gradient 
        z = b[i]  # switches off y on clipped samples
        for j in range(i):
            z = z - C[i, j] * y[j]
        z = z / C[i, i]
        e.append(ndtr(z)); s.append(ndtr(-z))
    E = torch.stack(e); S = torch.stack(s)
    Pprod = torch.cat([torch.ones(1, N), torch.cumprod(E[:-1], dim=0)])
    return (S * Pprod).sum(dim=0).mean()  # sum_i (1-e_i) prod_{j<i} e_j

# ---------- plastic mechanisms: margins and (beta_j, rho_jk) (torch) ----------
# M_{1,2} = sqrt(2) R1 +       R2            - F
# M_{2,3} =              R2 + sqrt(2) R3     - F
# M_{1,3} = (R1 + R3)/sqrt(2)                - F
C_PL = torch.tensor([[np.sqrt(2), 1.0, 0.0],
                     [0.0, 1.0, np.sqrt(2)],
                     [1/np.sqrt(2), 0.0, 1/np.sqrt(2)]])

def comps_p(a3, gv, cY=.1, cF=.1):
    """beta (m,) and rho (m,m) as differentiable torch expressions.
    Margins M_j = sum_i t_ji Y_i - F, Y_i ~ N(1,cY^2), F ~ N(1,cF^2)."""
    a3 = torch.clamp(a3, min=1e-6)
    R = (1 - gv) * a3
    t = C_PL * R  # t_ji = R_i |n_i.d_j|
    sM = torch.sqrt((cY * torch.linalg.norm(t, dim=1))**2 + cF**2)  # D[M_j]
    b = (t.sum(dim=1) - 1.0) / sM  # beta_j = E[M_j]/D[M_j]
    al = torch.cat([cY * t / sM[:, None],
                    -cF / sM[:, None]], dim=1)  # alpha_j in (Y1,Y2,Y3,F)
    rho = al @ al.T  # rho_jk = alpha_j.alpha_k
    return b, rho  # (|alpha_j| = 1; diag = 1)

# ---------- system beta: multinormal integral + autodiff ----------
# safe = all margins > 0 = {W <= beta}, W ~ N(0, rho), so pf = 1 - Phi_m
def beta_sys_torch(a3, gv):
    b, rho = comps_p(a3, gv)
    pf = genz_sf(b, rho)
    pc = torch.clamp(pf, 1e-300, 1 - 1e-16)
    bsys = -ndtri(pc)
    # pf >= max_i Phi(-b_i) exactly, so bsys <= min b_i; vs quad error
    return torch.minimum(bsys, b.min())

def beta_sys_grad(a3, gv):
    """Reverse-mode autodiff of a -> (beta, rho) -> pf -> beta_sys."""
    a = torch.tensor(np.asarray(a3, float), requires_grad=True)
    bs = beta_sys_torch(a, torch.tensor(np.asarray(gv, float)))
    g, = torch.autograd.grad(bs, a)
    return bs.item(), g.numpy()

# ---------- gradient-based RBDO with multi-start -------
def opt(gamma, x0s=([3., 3.], [1., 1.], [5., 5.]), bt=3.0):
    gv = np.array([gamma, 0, 0])
    V = lambda x: 2 * np.sqrt(2) * x[0] + x[1]
    dV = lambda x: np.array([2 * np.sqrt(2), 1.])
    def make_con(g):
        cache = {}
        def ev(x):  # one evaluation, both fun and jac
            key = np.asarray(x, float).tobytes()
            if key not in cache:
                bs, da = beta_sys_grad(np.array([x[0], x[1], x[0]]), g)
                cache[key] = (bs - bt, np.array([da[0] + da[2], da[1]]))  # a=(x0,x1,x0)
            return cache[key]
        return {'type': 'ineq', 'fun': lambda x: ev(x)[0], 'jac': lambda x: ev(x)[1]}, ev
    (c0, ev0), (c1, ev1) = make_con(np.zeros(3)), make_con(gv)
    best = None
    for x0 in x0s:
        r = minimize(V, x0, jac=dV, method='SLSQP', bounds=[(1e-3, 6.)] * 2,
                     constraints=[c0, c1], options={'ftol': 1e-12, 'maxiter': 300})
        if min(ev0(r.x)[0], ev1(r.x)[0]) > -1e-6 \
           and (best is None or r.fun < best.fun):
            best = r
    return best

# ---------- finite-difference check of the autodiff gradient ----------
def check_grads(eps=1e-6):
    rng = np.random.default_rng(0)
    err = 0.
    for gamma in (0., .5, .95):
        gv = np.array([gamma, 0, 0])
        for _ in range(5):
            a = rng.uniform(.5, 3., 3)
            bs, g = beta_sys_grad(a, gv)
            if bs > 4.5:  # deep tail: FD is unreliable, amplified by 1/phi(beta) 
                continue  
            fd = np.array([(beta_sys_grad(a + eps * e, gv)[0]
                            - beta_sys_grad(a - eps * e, gv)[0]) / (2 * eps)
                           for e in np.eye(3)])
            err = max(err, np.abs(g - fd).max() / max(np.abs(fd).max(), 1.))
    print(f"grad check plastic: max rel err {err:.2e}")

if __name__ == '__main__':
    check_grads()
    # plot graph gamma [0, 1]
    gs = np.concatenate([np.linspace(0, .95, 25), np.linspace(.96, .9999, 12)])
    t = time.time()
    xp = [3., 3.]; vs = []
    for g in gs:
        r = opt(g, x0s=[xp, [1., 1.], [5., 5.]])
        xp = r.x; vs.append(r.fun)
    vp1 = opt(1.0).fun
    print(f"sweep: {time.time()-t:.1f} s")
    print("plastic:", vs[0], vs[-1], vp1)

    vmax = max(max(vs), vp1)
    print(f"normalisation V*_max = {vmax:.6f}")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(gs, np.array(vs) / vmax, 'k-', label='plastic (system-level)')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'Damage $\gamma$')
    ax.set_ylabel(r'Optimum volume $V^*/V^*_{\max}$')
    ax.legend(); ax.grid(alpha=.3); plt.tight_layout()
    # plt.savefig('pfvol_gamma_plastic_genz.png', dpi=400)
    plt.show()
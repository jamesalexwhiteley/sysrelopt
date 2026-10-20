#type:ignore 
"""Elastic system-level definition."""
import numpy as np, time
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from truss_rbdo_bounds import comps_e, comps_p, pf_bounds, opt as opt_base

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 12})

# Author: James Whiteley (github.com/jamesalexwhiteley)

SQ2 = np.sqrt(2.)
NV = np.array([[1,-1],[1,0],[1,1]]) / np.array([SQ2,1,SQ2])[:,None]
LV = np.array([SQ2, 1., SQ2])
QP = np.array([1., SQ2])  # snapped member forces
GH_X, GH_W = np.polynomial.hermite.hermgauss(80)

def elastic_q(a1, a2, gamma):
    k = np.array([(1-gamma)*a1, a2, a1]) / LV
    K = sum(ki*np.outer(ni,ni) for ki,ni in zip(k, NV))
    return np.abs(k * (NV @ np.linalg.solve(K, [0,-1.])))

def pf_relaxed(a1, a2, gamma, cY=.1, cF=.1): 
    f = 1 + cF*SQ2*GH_X  # Gauss-Hermite nodes, F ~ N(1, cF^2)
    tB = QP / np.array([a2, a1])  # B_j = {Y_j <= tB_j f}, bars 2,3
    pB = norm.cdf((np.outer(f, tB) - 1)/cY)
    PB = 1 - (1-pB[:,0])*(1-pB[:,1])
    if gamma >= 1-1e-12:  # bar 1 gone: relaxed = remnant series
        return (GH_W/np.sqrt(np.pi)) @ PB
    Rb = np.array([(1-gamma)*a1, a2, a1])
    tA = elastic_q(a1, a2, gamma) / Rb  # A_i = {Y_i <= tA_i f}
    pA = norm.cdf((np.outer(f, tA) - 1)/cY)
    pU = norm.cdf((np.outer(f, np.maximum(tA[1:], tB)) - 1)/cY)  # A_i U B_i per bar
    PA  = 1 - (1-pA[:,0])*(1-pA[:,1])*(1-pA[:,2])
    PAB = 1 - (1-pA[:,0])*(1-pU[:,0])*(1-pU[:,1])
    return (GH_W/np.sqrt(np.pi)) @ (PA + PB - PAB)  # P((UA) N (UB) | f) integrated

def beta_und(x):  # undamaged elastic series system
    b, rho = comps_e(np.array([x[0],x[1],x[0]]), np.zeros(3))
    hi = np.clip(pf_bounds(b, rho)[1], 1e-300, 1-1e-16)
    return min(-norm.ppf(hi), b.min())

def beta_rel(x, gamma):
    pf = np.clip(pf_relaxed(x[0], x[1], gamma), 1e-300, 1-1e-16)
    bs = -norm.ppf(pf)
    if bs > -8: return bs
    # pf limited at 1: fall back to component beta (keeps gradient)
    bB = (np.array([x[1],x[0]])-QP)/np.hypot(.1*np.array([x[1],x[0]]), .1*QP)
    if gamma >= 1-1e-12: return bB.min()
    Rb = np.array([(1-gamma)*x[0],x[1],x[0]]); q = elastic_q(x[0],x[1],gamma)
    live = Rb > 1e-12
    bA = (Rb[live]-q[live])/np.hypot(.1*Rb[live], .1*q[live])
    return max(bA.min(), bB.min())

def opt_rel(gamma, x0s, bt=3.0):
    V = lambda x: 2*SQ2*x[0] + x[1]
    cons = [{'type':'ineq','fun': (lambda x: beta_und(x)-bt)},
            {'type':'ineq','fun': (lambda x, g=gamma: beta_rel(x,g)-bt)}]
    best = None
    for x0 in x0s:
        r = minimize(V, x0, method='SLSQP', bounds=[(1e-3,6.)]*2, constraints=cons,
                     options={'ftol':1e-12,'maxiter':300})
        if min(beta_und(r.x)-bt, beta_rel(r.x,gamma)-bt) > -1e-6 \
           and (best is None or r.fun < best.fun):
            best = r
    return best

if __name__ == '__main__':
    
    gs = np.concatenate([np.linspace(0,.95,25), np.linspace(.96,.9999,12)])
    t = time.time(); curves = {}; areas = {}
    for name, comps in [('elastic', comps_e), ('plastic', comps_p)]:
        xp=[3.,3.]; vs=[]; xs=[]
        for g in gs:
            r=opt_base(comps,g,x0s=[xp,[1.,1.],[5.,5.]]); xp=r.x
            vs.append(r.fun); xs.append(r.x)
        curves[name]=np.array(vs); areas[name]=np.array(xs)
    xp=[3.,3.]; vs=[]; xs=[]
    for g in gs:
        r=opt_rel(g,[xp,[1.,1.],[5.,5.]]); xp=r.x
        vs.append(r.fun); xs.append(r.x)
    curves['relaxed']=np.array(vs); areas['relaxed']=np.array(xs)
    r = opt_base(comps_e, 1.0);                        ve1, xe1 = r.fun, r.x
    r = opt_rel(1.0, [[2.5,1.5],[1.,1.],[5.,5.]]);     vr1, xr1 = r.fun, r.x
    ae, ap, ar = areas['elastic'], areas['plastic'], areas['relaxed']
    print(f"sweeps: {time.time()-t:.1f} s")
    print("elastic  :", curves['elastic'][0], curves['elastic'][-1], ve1)
    print("relaxed  :", curves['relaxed'][0], curves['relaxed'][-1], vr1)
    print("plastic  :", curves['plastic'][0], curves['plastic'][-1])

    from matplotlib.ticker import MaxNLocator
    amax = max(ae.max(), ar.max(), ap.max(), np.max(xe1), np.max(xr1))
    aticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(0, amax)

    for idx, lab, fname in [(0, r'Optimum area $a_1^*$', 'pfarea1_gamma_elastic_system.png'),
                            (1, r'Optimum area $a_2^*$', 'pfarea2_gamma_elastic_system.png')]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(gs, ae[:, idx], '-.', color='tab:blue',
                label=r'elastic (element-level), $\gamma<1$')
        ax.plot(1, xe1[idx], 'o', color='tab:blue', label=r'elastic, $\gamma=1$',
                clip_on=False, zorder=3)
        ax.plot(gs, ar[:, idx], ':', color='tab:blue', lw=2,
                label='elastic (system-level)')
        ax.plot(1, xr1[idx], 's', color='tab:blue', ms=4, clip_on=False, zorder=3)
        ax.plot(gs, ap[:, idx], 'k-', label='plastic (system-level)')
        ax.set_yticks(aticks); ax.set_ylim(0, aticks[-1])
        ax.set_xlim(0, 1); ax.margins(0)
        ax.set_xlabel(r'Damage $\gamma$')
        ax.set_ylabel(lab)
        ax.legend(); ax.grid(alpha=.3); plt.tight_layout()
        plt.savefig(fname, dpi=400)
        plt.close(fig)

    vmax = max(curves['elastic'].max(), ve1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(gs, curves['elastic']/vmax, '-.', color='tab:blue',
            label=r'elastic (element-level), $\gamma<1$')
    ax.plot(1, ve1/vmax, 'o', color='tab:blue', label=r'elastic, $\gamma=1$',
            clip_on=False, zorder=3)
    ax.plot(gs, curves['relaxed']/vmax, ':', color='tab:blue', lw=2,
            label='elastic (system-level)')
    ax.plot(1, vr1/vmax, 's', color='tab:blue', ms=4, clip_on=False, zorder=3)
    ax.plot(gs, curves['plastic']/vmax, 'k-', label='plastic (system-level)')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1); ax.margins(0)
    ax.set_xlabel(r'Damage $\gamma$')
    ax.set_ylabel(r'Optimum volume $V^*/V^*_{\max}$')
    # ax.set_title(r'$P_f$-constrained optimum vs damage ($\beta_{sys}\geq 3$)')
    ax.legend(); ax.grid(alpha=.3); plt.tight_layout()
    plt.savefig('pfvol_gamma_elastic_system.png', dpi=400)
    plt.show()
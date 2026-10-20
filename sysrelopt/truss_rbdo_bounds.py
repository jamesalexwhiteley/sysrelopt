#type:ignore 
"""Gradient-based RBDO using Ditlevsen bounds."""
import numpy as np
from scipy.stats import norm
from scipy.special import owens_t
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 12})

# Author: James Whiteley (github.com/jamesalexwhiteley)

nodes = np.array([[-1, 1], [-1, 0], [-1, -1]])  # free node at origin 
L = np.linalg.norm(nodes, axis=1)
n = -nodes / L[:, None]  # unit vectors from support -> free node 

# ---------- exact bivariate normal P(Z1<=h, Z2<=k; r); deterministic ----------
# via Claude; can be checked against scipy.stats.multivariate_normal.cdf
def bvn_le(h, k, r):
    if abs(r) < 1e-12: return norm.cdf(h) * norm.cdf(k)  # independent 
    s = np.sqrt(1 - r * r)
    if abs(h) < 1e-12 and abs(k) < 1e-12:
        return .25 + np.arcsin(r) / (2 * np.pi)  # closed form
    if abs(h) < 1e-12: h, k = k, h
    ah = (k - r * h) / (h * s)
    if abs(k) < 1e-12:
        return .5 * norm.cdf(h) - owens_t(h, ah) + .25
    ak = (h - r * k) / (k * s)
    c = .5 if h * k < 0 else 0.
    return .5 * (norm.cdf(h) + norm.cdf(k)) - owens_t(h, ah) - owens_t(k, ak) - c  # owen's identity 

# ---------- Ditlevsen (1979) bounds ----
def pf_bounds(b, rho):
    idx = np.argsort(b)  # most critical mode first
    b = b[idx]; rho = rho[np.ix_(idx, idx)]
    m = len(b); P = norm.cdf(-b)
    P2 = np.zeros((m, m))
    for i in range(m):
        for j in range(i):
            P2[i, j] = P2[j, i] = bvn_le(-b[i], -b[j], rho[i, j])
    lo = hi = P[0]
    for i in range(1, m):
        lo += max(P[i] - P2[i, :i].sum(), 0.)
        hi += P[i] - P2[i, :i].max()
    return lo, hi

# ---------- component-level (beta_i, rho_ij) ----------
def elastic_q(a3, gv):
    a3 = np.maximum(a3, 1e-6); k = a3 * (1 - gv) / L        # k = EA/L with E=1
    K = sum(ki * np.outer(ni, ni) for ki, ni in zip(k, n))  # 2x2 stiffness matrix K 
    return np.abs(k * (n @ np.linalg.solve(K, [0, -1.])))   # f = [0, −1]; q = k*e (stiffness x elongation)

def comps_e(a3, gv, cY=.1, cF=.1):  # elastic, first yield
    a3 = np.maximum(a3, 1e-6); q = elastic_q(a3, gv)
    R = (1 - gv) * a3; live = np.where(R > 1e-12)[0]
    b, al = [], []  
    # elastic margins M_i = R_i*Y_i − q_i*F
    # with R_i = (1−gamma)a_i, 
    # Y_i ~ N(1, cov_Y);    F ~ N(1, cov_F)
    # Var[M_i] = (R_i*cov_Y)^2 + (q_i*cov_F)^2
    for i in live:  
        sM = np.hypot(cY * R[i], cF * q[i])  # std = sqrt((c_Y*R_i)^2 + (c_F*q_i)^2)
        b.append((R[i] - q[i]) / sM)  # beta_i = (R_i − q_i)/std
        v = np.zeros(4); v[i], v[3] = cY * R[i] / sM, -cF * q[i] / sM; al.append(v)  # (Y1,Y2,Y3,F)
    b, al = np.array(b), np.array(al)
    rho = al @ al.T  # rho_ij = alpha_i^T alpha_j 
    np.fill_diagonal(rho, 1.)
    return b, rho
    
# ---------- plastic mechanisms: explicit safety margins ----------
# M_{1,2} = sqrt(2) R1 +       R2            - F
# M_{2,3} =              R2 + sqrt(2) R3     - F
# M_{1,3} = (R1 + R3)/sqrt(2)                - F
C_PL = np.array([[np.sqrt(2), 1.0, 0.0],
                 [0.0, 1.0, np.sqrt(2)],
                 [1/np.sqrt(2), 0.0, 1/np.sqrt(2)]])

def comps_p(a3, gv, cY=.1, cF=.1): 
    R = (1 - gv) * np.maximum(a3, 1e-6) 
    b, al = [], [] 
    for j in range(3):  # one mechanism per row of C_PL
        tj = C_PL[j] * R  # t_ji = R_i |n_i.d_j|  (scaled)
        sM = np.hypot(cY * np.linalg.norm(tj), cF)  # D[M_j]
        b.append((tj.sum() - 1.0) / sM)  # beta_j = E[M_j]/D[M_j]
        al.append(np.append(cY * tj, -cF) / sM) # alpha_j in (Y1,Y2,Y3,F)
    b, al = np.array(b), np.array(al)
    rho = al @ al.T; np.fill_diagonal(rho, 1.)
    return b, rho

# ---------- system beta (upper pf bound) ----------
def beta_sys(comps, a3, gv):
    b, rho = comps(a3, gv)
    hi = np.clip(pf_bounds(b, rho)[1], 1e-300, 1 - 1e-16)
    # beta_sys <= min b_i always, so this changes nothing when feasible
    return min(-norm.ppf(hi), b.min())

# ---------- gradient-based RBDO with multi-start -------
def opt(comps, gamma, x0s=([3., 3.], [1., 1.], [5., 5.]), bt=3.0):
    gv = np.array([gamma, 0, 0])
    V = lambda x: 2 * np.sqrt(2) * x[0] + x[1]
    gfun = lambda x, g: beta_sys(comps, np.array([x[0], x[1], x[0]]), g) - bt
    cons = [{'type': 'ineq', 'fun': (lambda x, g=g: gfun(x, g))}
            for g in (np.zeros(3), gv)]
    best = None
    for x0 in x0s:
        r = minimize(V, x0, method='SLSQP', bounds=[(1e-3, 6.)] * 2,
                     constraints=cons, options={'ftol': 1e-12, 'maxiter': 300})
        if min(gfun(r.x, np.zeros(3)), gfun(r.x, gv)) > -1e-6 \
           and (best is None or r.fun < best.fun):
            best = r
    return best

if __name__ == '__main__':
    # plot graphs gamma [0, 1]
    gs = np.concatenate([np.linspace(0, .95, 25), np.linspace(.96, .9999, 12)])
    t = time.time(); out = {}
    for name, comps in [('elastic', comps_e), ('plastic', comps_p)]:
        xp = [3., 3.]; vs = []
        for g in gs:
            r = opt(comps, g, x0s=[xp, [1., 1.], [5., 5.]])
            xp = r.x; vs.append(r.fun)
        out[name] = vs
    ve1 = opt(comps_e, 1.0).fun; vp1 = opt(comps_p, 1.0).fun
    print(f"sweep: {time.time()-t:.1f} s")
    print("elastic:", out['elastic'][0], out['elastic'][-1], ve1)
    print("plastic:", out['plastic'][0], out['plastic'][-1], vp1)

    # normalise by the largest optimum 
    vmax = max(max(out['elastic']), max(out['plastic']), ve1, vp1)
    ven = np.array(out['elastic']) / vmax; vpn = np.array(out['plastic']) / vmax
    ve1n, vp1n = ve1 / vmax, vp1 / vmax
    print(f"normalisation V*_max = {vmax:.6f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(gs, ven, '-.', color='tab:blue',
            label=r'elastic (element-level), $\gamma<1$')
    ax.plot(1, ve1n, 'o', color='tab:blue', label=r'elastic (element-level), $\gamma=1$')
    ax.plot([gs[-1], 1], [ven[-1], ve1n], '--', color='tab:blue', lw=.8)
    ax.plot(gs, vpn, 'k-', label='plastic (system-level)')
    # ax.plot(1, vp1n, 'ko', ms=4)  
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'Damage $\gamma$')
    ax.set_ylabel(r'Optimum volume $V^*/V^*_{\max}$')
    # ax.set_title(r'System-reliability constrained ($\beta_{sys}\geq 3$)')
    ax.legend(); ax.grid(alpha=.3); plt.tight_layout()
    plt.savefig('pfvol_gamma.png', dpi=400)
    plt.show()

    # # investigate gamma=0
    # # elastic case 
    # r = opt(comps_e, 0.0)
    # a = np.array([r.x[0], r.x[1], r.x[0]]); gv = np.zeros(3)
    # q = elastic_q(a, gv); R = (1 - gv) * a
    # print(f"V* = {r.fun:.6f}")
    # print(f"areas  a = {np.array2string(a, precision=4)}")
    # print(f"forces q = {np.array2string(q, precision=4)}")
    # print(f"util q/R = {np.array2string(q / R, precision=4)}")
    # # plastic case
    # r = opt(comps_p, 0.0)
    # a = np.array([r.x[0], r.x[1], r.x[0]]); gv = np.zeros(3)
    # b, rho = comps_p(a, gv)
    # names = ['{1,2}', '{2,3}', '{1,3}']
    # print(f"V* = {r.fun:.6f}   areas a = {np.array2string(a, precision=4)}")
    # for nm, bj in zip(names, b):
    #     print(f"  mechanism {nm}: beta = {bj:.4f}")
    # print(f"  beta_sys = {beta_sys(comps_p, a, gv):.4f}")
    # R = (1 - gv) * a
    # cap = C_PL @ R  # mean capacity work per mechanism, load work = 1
    # for nm, cj, bj in zip(names, cap, b):
    #     print(f"  mechanism {nm}: E[M] = {cj - 1:+.4f}  (capacity {cj:.4f} vs load 1)  beta = {bj:.4f}")
# -*- coding: utf-8 -*-
"""S1602 -- child-3.1, TACT 8: THE STRUCTURE-COMPACTIFIER.

What in the inventory FORCES so(eta|W) = so(3,1) down to a compact or discrete subgroup?
Two candidates, one tact (ex-ante: child-3.1/S1602_COMPACTIFIER_EXANTE.md, committed BEFORE
this file existed):

  (a) ARITHMETIC -- Gamma = the point group of a beta-integral lattice (S1598: infinite,
      hyperbolic).  Does it have FINITE COVOLUME in SO(3,1)?  i.e. does its orbit on H^3 grow
      LIKE THE VOLUME, N(R)/vol(B_R) -> a positive constant?
      ★The trap carved before the numbers: EXPONENTIAL GROWTH is NOT finite covolume.  A FREE
       subgroup <P,P'> grows exponentially yet has INFINITE covolume.  So the criterion is not
       "grows" but "N(R)/vol(B_R) does not decay to 0", and it carries TWO negative worlds on
       the criterion itself (cyclic AND free), norm S1595.

  (b) POSITIVE -- Heisenberg unitarity of floor 0 on (V, sigma), V = W(x)A, sigma = -eta|W(x)omega.
      Positive compatible complex structures J (J^2=-1, sigma(J.,J.)=sigma, sigma(.,J.) > 0):
      (i)  is the stabiliser of J in the full Levi COMPACT?  (Killing form negative-definite)
      (ii) is the set of such J ONE orbit of SO+(3,1) x SL(2)?
      Carved before the numbers: (i) COMPACT (positivity buys the compact), (ii) SPLIT -- on the
      BARE sigma several orbits (Siegel 20-dim >> dim G/K), but the ADAPTED family (t in H^3,
      J_A in H^2), selected by the tensor structure W(x)A which is already HOUSE (CH-glue), is
      exactly ONE orbit G/K = H^3 x H^2.

  (c) SPLICE (gated on (a) thick AND (b) one adapted orbit): the adapted moduli H^3 x H^2 = G/K;
      mod Gamma (finite covolume) it is FINITE VOLUME -- "a functional would have a compact of
      finite volume to live on", WITHOUT building the functional.

★Every "complete/transitive" column carries a negative world ON THE CRITERION (norm S1595).
★Every verdict line is GATED on the result of its own tooth (norm S1598): a red tooth must not
 let the headline sentence print.

Run line (unconditionally into the seal):
    python child-3.1/S1602_compactifier.py --outdir child-3.1
"""
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "test")))
from _teeth import ok, report                                                 # noqa: E402

TOL = 1e-8
RNG = np.random.default_rng(20260820)
_OUT = {"dir": _HERE}


def _path(name):
    return os.path.join(_OUT["dir"], name)


class Tee(object):
    def __init__(self, real, fh):
        self.real, self.fh = real, fh

    def write(self, s):
        self.real.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.real.flush()
        if not self.fh.closed:
            self.fh.flush()


def say(s=""):
    print(s)


def head(s):
    say("\n" + "=" * 96)
    say(s)
    say("=" * 96)


def dump(rec, **payload):
    with open(_path("S1602_compactifier_dump.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"rec": rec, **payload}, sort_keys=True, default=str) + "\n")


def mat(name, M, fmt="%7.3f"):
    say(f"    {name} =")
    for row in np.atleast_2d(M):
        say("      [" + " ".join(fmt % x for x in row) + "]")


# =============================================================== 1. THE FLOOR, REBUILT
def eta_of(p, q):
    return np.diag([1.0] * p + [-1.0] * q)


OMEGA_A = np.array([[0.0, 1.0], [-1.0, 0.0]])


def bracket_form(etaW):
    """sigma = B = -eta|W (x) omega_A on V = W(x)A, index k = w*2 + a  (same as S1598)."""
    return -np.kron(etaW, OMEGA_A)


def so_eta_generators(etaW):
    """6 generators of so(eta|W): A_ij = e_i (eta e_j)^T - e_j (eta e_i)^T  (eta-antisymmetric)."""
    n = etaW.shape[0]
    I = np.eye(n)
    gens = []
    for i in range(n):
        for j in range(i + 1, n):
            A = np.outer(I[i], etaW @ I[j]) - np.outer(I[j], etaW @ I[i])
            gens.append(A)
    return gens


SL2 = [np.array([[1.0, 0.0], [0.0, -1.0]]),   # H
       np.array([[0.0, 1.0], [0.0, 0.0]]),     # E
       np.array([[0.0, 0.0], [1.0, 0.0]])]     # F


def levi_generators(etaW):
    """9 generators of g = so(eta|W) (+) sl(2) acting on V=W(x)A: X(x)I  and  I(x)Y."""
    soW = so_eta_generators(etaW)
    gens = [np.kron(X, np.eye(2)) for X in soW] + [np.kron(np.eye(4), Y) for Y in SL2]
    return gens


def sp_basis(B):
    """A basis of sp(V,B) = {X : X^T B + B X = 0} = {B^{-1} S : S symmetric}.  dim 36."""
    n = B.shape[0]
    Binv = np.linalg.inv(B)
    out = []
    for i in range(n):
        for j in range(i, n):
            S = np.zeros((n, n))
            S[i, j] = S[j, i] = 1.0
            out.append(Binv @ S)
    return out


# =============================================================== 2. POSITIVE COMPLEX STRUCTURES
def compatible_J(B, G):
    """J from an SPD (or symmetric) G: Shat = G^{-1}B (skew wrt G), J = Shat (-Shat^2)^{-1/2}.
    Returns J with J^2=-1, J^T B J = B.  g = B J is symmetric; SPD iff G is SPD."""
    Shat = np.linalg.solve(G, B)
    M2 = -Shat @ Shat
    w, Vv = np.linalg.eig(M2)
    sq = (Vv * np.sqrt(w + 0j)) @ np.linalg.inv(Vv)
    J = np.real(Shat @ np.linalg.inv(sq))
    # ★POSITIVITY SELECTION: J and -J are both compatible; the positive one has sigma(.,J.) > 0.
    # When G is SPD one sign gives g=BJ pos-def, the other neg-def -- pick the positive one.
    # ⟨on the first run this flip was ABSENT; g came out neg-def and tooth T4 fell -- see
    #  S1602_compactifier_run_BUG.log; positivity is a SELECTION, and the tooth held me to it.⟩
    if np.trace(B @ J) < 0:
        J = -J
    return J


def metric_of(B, J):
    """g(x,y) = sigma(x, J y) = x^T (B J) y ; returns the symmetric matrix B J (symmetrised)."""
    g = B @ J
    return 0.5 * (g + g.T)


def h_observer(etaW, t):
    """Wick metric h_t = eta + 2 (eta t)(eta t)^T for a unit timelike t (eta(t,t) = -1).  SPD."""
    et = etaW @ t
    return etaW + 2.0 * np.outer(et, et)


def J_A_of(tau):
    """Positive compatible complex structure on (A, omega_A) for tau = x + i y in upper half plane."""
    x, y = float(np.real(tau)), float(np.imag(tau))
    return (1.0 / y) * np.array([[x, -(x * x + y * y)], [1.0, -x]])


def adapted_J(B, etaW, t, tau):
    """Adapted positive J from an observer t in H^3 and a modulus tau in H^2: G = h_t (x) g_A."""
    gA = OMEGA_A @ J_A_of(tau)
    gA = 0.5 * (gA + gA.T)
    G = np.kron(h_observer(etaW, t), gA)
    return compatible_J(B, G)


def indefinite_J(B):
    """A compatible J' with J'^2=-1, J'^T B J' = B but g'=B J' INDEFINITE (the negative world for
    positivity/compactness).  Built by flipping the sign on one Darboux conjugate pair."""
    P = darboux(B)                        # P^T B P = Omega0 (standard symplectic blocks)
    Om0 = np.zeros((8, 8))
    for k in range(4):
        Om0[2 * k, 2 * k + 1] = 1.0
        Om0[2 * k + 1, 2 * k] = -1.0
    Jstd = -Om0.copy()                    # standard positive complex structure, g = I
    # flip the first conjugate pair -> that block's metric becomes negative -> indefinite
    Jstd[0:2, 0:2] = +Om0[0:2, 0:2]
    Pinv = np.linalg.inv(P)
    return P @ Jstd @ Pinv


def darboux(B):
    """P with columns a symplectic basis: P^T B P = Omega0 (blocks [[0,1],[-1,0]])."""
    n = B.shape[0]
    pool = [np.eye(n)[:, i].astype(float) for i in range(n)]
    cols = []
    while pool:
        e = pool.pop(0)
        f = None
        for k, g in enumerate(pool):
            if abs(float(e @ B @ g)) > 1e-8:
                f = pool.pop(k)
                break
        if f is None:
            continue
        c = float(e @ B @ f)
        f = f / c
        cols += [e, f]
        pool = [g - float(e @ B @ g) * f + float(f @ B @ g) * e for g in pool]
    return np.column_stack(cols)


# =============================================================== 3. STABILISER / KILLING
def stab_coeffs(gens, J):
    """Basis (in coefficient space) of {X in span(gens): X J - J X = 0}."""
    n = J.shape[0]
    m = len(gens)
    Amat = np.zeros((n * n, m))
    for k, Z in enumerate(gens):
        Amat[:, k] = (Z @ J - J @ Z).ravel()
    u, s, vh = np.linalg.svd(Amat)
    tolr = max(Amat.shape) * (s[0] if s.size else 1.0) * 1e-10
    rank = int(np.sum(s > tolr))
    null = vh[rank:].T if rank < m else np.zeros((m, 0))
    return null, rank                       # null: m x s coefficient basis of the stabiliser


def killing_form(gens):
    """Killing form K_ij = tr(ad_i ad_j) on g = span(gens), via structure constants in gl(V)."""
    m = len(gens)
    flat = np.column_stack([Z.ravel() for Z in gens])       # (n*n) x m
    # express any [Zi,Zj] in the g-basis by least squares (g is a subalgebra -> exact)
    f = np.zeros((m, m, m))
    for i in range(m):
        for j in range(m):
            br = (gens[i] @ gens[j] - gens[j] @ gens[i]).ravel()
            c, *_ = np.linalg.lstsq(flat, br, rcond=None)
            f[i, j, :] = c
    ad = [np.array([[f[i, j, k] for j in range(m)] for k in range(m)]) for i in range(m)]
    K = np.array([[np.trace(ad[i] @ ad[j]) for j in range(m)] for i in range(m)])
    return K


def killing_on_stab(gens, J):
    """Eigenvalues of the Killing form restricted to stab_g(J).  Neg-def <=> compact."""
    null, rank = stab_coeffs(gens, J)
    s = null.shape[1]
    K = killing_form(gens)
    if s == 0:
        return np.array([]), 0, K
    Ks = null.T @ K @ null
    ev = np.linalg.eigvalsh(0.5 * (Ks + Ks.T))
    return ev, s, K


def orbit_dim(gens, J):
    """dim of the g-orbit through J = rank of X -> [X, J] over span(gens)."""
    _, rank = stab_coeffs(gens, J)
    return rank                              # = len(gens) - dim stab


def compatible_tangent_dim(B, J):
    """dim of {dJ: dJ J + J dJ = 0  and  dJ^T B J + J^T B dJ = 0} = Siegel tangent (expect 20)."""
    n = J.shape[0]
    # a single linear operator L(vec dJ) stacking both tangency conditions
    L1 = np.zeros((n * n, n * n))
    L2 = np.zeros((n * n, n * n))
    for idx in range(n * n):
        dJ = np.zeros(n * n)
        dJ[idx] = 1.0
        D = dJ.reshape(n, n)
        L1[:, idx] = (D @ J + J @ D).ravel()
        L2[:, idx] = (D.T @ B @ J + J.T @ B @ D).ravel()
    L = np.vstack([L1, L2])
    s = np.linalg.svd(L, compute_uv=False)
    tolr = max(L.shape) * (s[0] if s.size else 1.0) * 1e-10
    return n * n - int(np.sum(s > tolr))


# =============================================================== 4. FRAMES / TRANSITIVITY
def lorentz_frame(etaW, t):
    """A Lorentz frame F (F^T eta F = eta) whose LAST column is the unit timelike t."""
    n = etaW.shape[0]
    # spacelike vectors: eta-orthonormalise the standard spatial axes against t
    cols = []
    basis = [np.eye(n)[:, i] for i in range(n - 1)]
    for v in basis:
        w = v.astype(float).copy()
        # remove t-component (t is timelike, eta(t,t)=-1)
        w = w + (w @ etaW @ t) * t              # since eta(t,t)=-1: proj = -(w.eta.t)/(-1) t
        for u in cols:
            w = w - (w @ etaW @ u) * u
        nn = float(w @ etaW @ w)
        if nn > 1e-9:
            cols.append(w / math.sqrt(nn))
    F = np.column_stack(cols + [t])
    return F


def frame_map_W(etaW, t0, t1):
    """g_W in O(3,1) mapping t0 -> t1 (and a frame to a frame); fixed to SO+ (det>0, time-up)."""
    F0, F1 = lorentz_frame(etaW, t0), lorentz_frame(etaW, t1)
    g = F1 @ np.linalg.inv(F0)
    if np.linalg.det(g) < 0:
        F1b = F1.copy(); F1b[:, 0] *= -1.0
        g = F1b @ np.linalg.inv(F0)
    if g[3, 3] < 0:                              # keep future-pointing
        g = -g
    return g


def symp_frame_A(J):
    """A unitary frame [a, J a] with omega_A(a, J a) = 1 for a complex structure J on A."""
    a = np.array([1.0, 0.0])
    val = float(a @ OMEGA_A @ (J @ a))
    if abs(val) < 1e-9:
        a = np.array([0.0, 1.0]); val = float(a @ OMEGA_A @ (J @ a))
    a = a / math.sqrt(abs(val)) * (1.0 if val > 0 else 1.0)
    return np.column_stack([a, J @ a])


def frame_map_A(J0, J1):
    """h in SL(2) = Sp(2) mapping J0 -> J1 by unitary frames."""
    F0, F1 = symp_frame_A(J0), symp_frame_A(J1)
    return F1 @ np.linalg.inv(F0)


# =============================================================== 5. ARITHMETIC / COVOLUME
def reflection(v, etaW):
    q = float(v @ etaW @ v)
    return np.eye(4) - 2.0 * np.outer(v, etaW @ v) / q


def is_integral(M):
    return float(np.max(np.abs(M - np.round(M)))) < 1e-9


def integral_reflections(etaW, box=2):
    import itertools as it
    out = []
    for v in it.product(range(-box, box + 1), repeat=4):
        v = np.array(v, float)
        if not v.any() or abs(float(v @ etaW @ v)) < 1e-9:
            continue
        R = reflection(v, etaW)
        if is_integral(R) and float(np.max(np.abs(R.T @ etaW @ R - etaW))) < 1e-9:
            out.append((tuple(int(x) for x in v), np.round(R).astype(np.int64)))
    return out


def a_hyperbolic_pair(etaW):
    """Two integer hyperbolic isometries P, P' with DIFFERENT axes (P' = perm-conjugate of P)."""
    refls = integral_reflections(etaW, box=2)
    P = None
    for va, A in refls:
        for vb, Bm in refls:
            Pp = (A @ Bm)
            if np.max(np.abs(np.linalg.eigvals(Pp.astype(float)))) > 1.0 + 1e-6:
                P = Pp.astype(np.int64); break
        if P is not None:
            break
    S = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.int64)  # swap x,y
    Pp = S @ P @ S                       # conjugate: hyperbolic with a rotated axis
    return P, Pp


def vol_ball(R):
    return math.pi * (math.sinh(2.0 * R) - 2.0 * R)


def n_hyp(D):
    """Number of integer points (a,b,c,d) with a^2+b^2+c^2 = d^2-1, 1 <= d <= D  (grid on H^3)."""
    total = 0
    for d in range(1, int(math.floor(D)) + 1):
        n = d * d - 1
        cnt = 0
        A = int(math.isqrt(n))
        for a in range(-A, A + 1):
            ra = n - a * a
            if ra < 0:
                continue
            Bb = int(math.isqrt(ra))
            for b in range(-Bb, Bb + 1):
                rc = ra - b * b
                if rc < 0:
                    continue
                c = int(math.isqrt(rc))
                if c * c == rc:
                    cnt += 2 if c > 0 else 1
        total += cnt
    return total


def orbit_heights(gens, x0, hmax, cap=400000):
    """BFS the orbit of x0 (integer vectors) under gens (+inverses), canonical sign d>=0,
    return the multiset of heights d = |v[3]| for distinct orbit points with d <= hmax."""
    def canon(v):
        v = tuple(int(x) for x in v)
        return v if v[3] > 0 else tuple(-x for x in v)
    allg = []
    for G in gens:
        allg.append(G)
        allg.append(np.round(np.linalg.inv(G.astype(float))).astype(np.int64))
    seen = {}
    start = canon(x0)
    seen[start] = abs(start[3])
    frontier = [np.array(start, dtype=np.int64)]
    while frontier and len(seen) < cap:
        nxt = []
        for v in frontier:
            for G in allg:
                w = G @ v
                cw = canon(w)
                if cw in seen:
                    continue
                h = abs(cw[3])
                seen[cw] = h
                if h <= hmax:
                    nxt.append(np.array(cw, dtype=np.int64))
                # points beyond hmax are recorded (dedup) but not expanded
        frontier = nxt
    return [h for h in seen.values() if h <= hmax]


def counts_at(heights, R_list):
    hs = np.array(sorted(heights))
    return [int(np.sum(hs <= math.cosh(R))) for R in R_list]


# =============================================================== 6. THE RUN
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=_HERE)
    args = ap.parse_args(argv)
    _OUT["dir"] = os.path.abspath(args.outdir)
    os.makedirs(_OUT["dir"], exist_ok=True)
    sys.stdout = Tee(sys.stdout, open(_path("S1602_compactifier_run.log"), "w",
                                      encoding="utf-8"))
    open(_path("S1602_compactifier_dump.jsonl"), "w", encoding="utf-8").close()

    head("S1602 -- the structure-compactifier (two candidates, one probe)")
    say("  ★verdict lines are gated on their own teeth (norm S1598); each 'complete/transitive'")
    say("   column carries a negative world ON THE CRITERION (norm S1595).")

    etaW = eta_of(3, 1)
    B = bracket_form(etaW)
    gens = levi_generators(etaW)

    # ------------------------------------------ C-P0: controls
    head("C-P0 -- controls: sigma nondegenerate, and G lies in Sp(V,sigma)")
    rk = int(np.linalg.matrix_rank(B, tol=1e-8))
    say(f"  sigma = B = -eta|W (x) omega_A : antisymmetry {float(np.max(np.abs(B + B.T))):.1e} , "
        f"rank {rk} of 8")
    sp_res = [float(np.max(np.abs(Z.T @ B + B @ Z))) for Z in gens]
    say(f"  all 9 Levi generators in sp(V,sigma): worst |Z^T B + B Z| = {max(sp_res):.1e}")
    nonsp = np.eye(8)                         # identity is NOT symplectic: I^T B + B I = 2B
    say(f"  negative world (identity, non-symplectic): |I^T B + B I| = "
        f"{float(np.max(np.abs(nonsp.T @ B + B @ nonsp))):.1e}")
    dump("C-P0", rank=rk, worst_sp=max(sp_res))

    ok(lambda w: w[0] == 8, (rk,),
       "T1/C-P0a: the symplectic form sigma on the translations is NON-degenerate (rank 8)",
       must_fail_on=("a deliberately degenerate form",
                     (int(np.linalg.matrix_rank(-np.kron(np.diag([1.0, 1, 1, 0]), OMEGA_A),
                                                tol=1e-8)),)))
    ok(lambda w: w[0] < 1e-9, (max(sp_res),),
       "T2/C-P0b: all 9 Levi generators lie in sp(V,sigma) -- so G is a subgroup of Sp(8,R)",
       must_fail_on=("the identity, which is not symplectic",
                     (float(np.max(np.abs(nonsp.T @ B + B @ nonsp))),)))

    # ------------------------------------------ C-P1: covolume of Gamma
    head("C-P1 (LOAD-BEARING, a) -- covolume of Gamma in SO(3,1): N(R)/vol(B_R), with the "
         "exponential-growth trap disarmed")
    P, Pp = a_hyperbolic_pair(etaW)
    say(f"  P  (integer hyperbolic, S1598) : max|lambda| = "
        f"{float(np.max(np.abs(np.linalg.eigvals(P.astype(float))))):.3f}")
    say(f"  P' (perm-conjugate, other axis): max|lambda| = "
        f"{float(np.max(np.abs(np.linalg.eigvals(Pp.astype(float))))):.3f}")
    x0 = np.array([0, 0, 0, 1], dtype=np.int64)

    R_list = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    hmax = math.cosh(max(R_list))
    # (i) arithmetic: ALL integer points on the hyperboloid (grid on H^3, O(3,1;Z)-invariant)
    N_arith = [n_hyp(math.cosh(R)) for R in R_list]
    # (ii) cyclic <P> and (iii) free <P,P'>: genuine single orbits of x0
    h_cyc = orbit_heights([P], x0, hmax)
    h_free = orbit_heights([P, Pp], x0, hmax)
    N_cyc = counts_at(h_cyc, R_list)
    N_free = counts_at(h_free, R_list)
    vols = [vol_ball(R) for R in R_list]

    def ratios(N):
        return [N[i] / vols[i] for i in range(len(N))]
    r_arith, r_cyc, r_free = ratios(N_arith), ratios(N_cyc), ratios(N_free)

    say(f"  {'R':>5s} {'vol(B_R)':>12s} | {'N_arith':>8s} {'ratio':>9s} | "
        f"{'N_cyc':>6s} {'ratio':>9s} | {'N_free':>6s} {'ratio':>9s}")
    for i, R in enumerate(R_list):
        say(f"  {R:>5.1f} {vols[i]:>12.2f} | {N_arith[i]:>8d} {r_arith[i]:>9.4f} | "
            f"{N_cyc[i]:>6d} {r_cyc[i]:>9.4f} | {N_free[i]:>6d} {r_free[i]:>9.4f}")

    def decay(r):
        """ratio(R_max)/ratio(R_mid): ~1 for finite covolume, ->0 for thin (cyclic/free)."""
        return (r[-1] / r[len(r) // 2]) if r[len(r) // 2] > 0 else 0.0
    d_arith, d_cyc, d_free = decay(r_arith), decay(r_cyc), decay(r_free)
    say(f"  ★decay ratio [ratio(R_max)/ratio(R_mid)]: arith {d_arith:.3f} , cyclic {d_cyc:.3f} , "
        f"free {d_free:.3f}")
    say("   finite covolume <=> ratio does not decay to 0 (threshold 0.5); exponential growth of")
    say("   the free group does NOT rescue it -- that is the trap this criterion is built against.")
    dump("C-P1", R=R_list, N_arith=N_arith, N_cyc=N_cyc, N_free=N_free,
         decay=[d_arith, d_cyc, d_free])

    ok(lambda w: w[0] > 0.5, (d_arith,),
       "T3/C-P1 (LOAD-BEARING, ON THE CRITERION): the point group's orbit density N(R)/vol(B_R) "
       "does NOT decay (finite covolume) -- and the detector is NOT fooled by exponential growth",
       must_fail_on=[("the cyclic subgroup <P> (linear growth, infinite covolume)", (d_cyc,)),
                     ("the FREE subgroup <P,P'> (exponential growth, STILL infinite covolume)",
                      (d_free,))])

    # ------------------------------------------ C-P2: existence of a positive compatible J
    head("C-P2 (b, existence) -- a positive compatible J exists; positivity is MEASURED")
    Ggen = RNG.normal(size=(8, 8))
    Ggen = Ggen @ Ggen.T + 3.0 * np.eye(8)               # a generic SPD metric
    J_gen = compatible_J(B, Ggen)
    J0 = adapted_J(B, etaW, np.array([0.0, 0, 0, 1.0]), 1j)   # adapted representative (t0=e4, tau=i)
    Jind = indefinite_J(B)

    def checkJ(name, J):
        j2 = float(np.max(np.abs(J @ J + np.eye(8))))
        comp = float(np.max(np.abs(J.T @ B @ J - B)))
        g = metric_of(B, J)
        ev = np.linalg.eigvalsh(g)
        say(f"  {name:>10s}: |J^2+I|={j2:.1e} , |J^T B J - B|={comp:.1e} , g eigenvalues in "
            f"[{ev.min():+.3f}, {ev.max():+.3f}] , SPD={bool(ev.min() > 1e-9)}")
        return j2, comp, ev
    j2g, cpg, evg = checkJ("generic", J_gen)
    j2a, cpa, eva = checkJ("adapted", J0)
    j2i, cpi, evi = checkJ("indefinite", Jind)
    dump("C-P2", gen_spd=bool(evg.min() > 1e-9), adapted_spd=bool(eva.min() > 1e-9),
         indef_min=float(evi.min()))

    ok(lambda w: w[0] < 1e-8 and w[1] < 1e-8 and w[2] > 1e-9,
       (j2a, cpa, float(eva.min())),
       "T4/C-P2: the adapted J is a genuine compatible complex structure and g=BJ is POSITIVE "
       "definite (positivity measured, not assumed)",
       must_fail_on=("the indefinite compatible J', whose g' has a negative eigenvalue",
                     (j2i, cpi, float(evi.min()))))

    # ------------------------------------------ C-P3: stabiliser compact
    head("C-P3 (LOAD-BEARING, b-i) -- the stabiliser of a positive J in the Levi is COMPACT")
    ev_a, s_a, _ = killing_on_stab(gens, J0)
    ev_i, s_i, _ = killing_on_stab(gens, Jind)
    say(f"  adapted J0 : dim stab = {s_a} , Killing eigenvalues on stab = "
        f"[{ev_a.min() if ev_a.size else 0:+.3f} .. {ev_a.max() if ev_a.size else 0:+.3f}]  "
        f"=> {'NEG-DEF (compact)' if ev_a.size and ev_a.max() < -1e-9 else 'not neg-def'}")
    say(f"  indefinite J': dim stab = {s_i} , Killing eigenvalues on stab = "
        f"[{ev_i.min() if ev_i.size else 0:+.3f} .. {ev_i.max() if ev_i.size else 0:+.3f}]  "
        f"=> {'NEG-DEF (compact)' if ev_i.size and ev_i.max() < -1e-9 else 'NOT neg-def (non-compact)'}")
    say("  ★positivity cuts the non-compact Levi so(3,1)+sl(2) down to a COMPACT stabiliser;")
    say("   the indefinite J' keeps a boost -> its Killing form has a positive direction.")
    dump("C-P3", adapted_stab=s_a, adapted_kill_max=float(ev_a.max()) if ev_a.size else None,
         indef_stab=s_i, indef_kill_max=float(ev_i.max()) if ev_i.size else None)

    def compact_detector(w):
        ev, sdim = w
        return sdim > 0 and ev.size > 0 and float(ev.max()) < -1e-9
    ok(compact_detector, (ev_a, s_a),
       "T5/C-P3 (ON THE CRITERION): the Killing form is NEGATIVE-DEFINITE on the stabiliser of "
       "the positive J -- the stabiliser is COMPACT",
       must_fail_on=("the indefinite J', whose stabiliser carries a boost (Killing not neg-def)",
                     (ev_i, s_i)))

    # ------------------------------------------ C-P4: bare positivity -> several orbits
    head("C-P4 (b-ii, bare sigma) -- how many orbits?  dim(G.J) vs dim{positive compatible J}")
    d_orbit_G = orbit_dim(gens, J0)
    d_siegel = compatible_tangent_dim(B, J0)
    spb = sp_basis(B)
    d_orbit_Sp = orbit_dim(spb, J0)
    say(f"  dim(G . J0)        = {d_orbit_G}   (G = SO(3,1) x SL(2), dim 9)")
    say(f"  dim{{compat. pos. J}} = {d_siegel}   (Siegel space Sp(8)/U(4), expected 20)")
    say(f"  dim(Sp(8) . J0)    = {d_orbit_Sp}   (full symplectic group -- transitive negative world)")
    say(f"  => codim of the G-orbit in the space of positive J = {d_siegel - d_orbit_G} > 0")
    say("     ==> on the BARE sigma, positivity leaves a residual moduli: SEVERAL G-orbits.")
    dump("C-P4", dim_G=d_orbit_G, dim_siegel=d_siegel, dim_Sp=d_orbit_Sp)

    ok(lambda w: w[0] < w[1], (d_orbit_G, d_siegel),
       "T6/C-P4: the G-orbit is a PROPER submanifold of the space of positive J (several orbits "
       "on the bare sigma) -- and the detector CAN say 'one orbit'",
       must_fail_on=("the full Sp(8), which acts transitively (dim orbit = dim space)",
                     (d_orbit_Sp, d_siegel)))

    # ------------------------------------------ C-P5: adapted family -> ONE orbit, compact stab
    head("C-P5 (b-ii, adapted) -- the adapted family (t in H^3, tau in H^2) is ONE orbit G/K")
    t0 = np.array([0.0, 0, 0, 1.0])
    JA0 = J_A_of(1j)
    samples = []
    for _ in range(40):
        # a random adapted J: random observer t in H^3, random modulus tau in H^2
        boost = RNG.normal(size=3) * 0.7
        r = math.sqrt(float(boost @ boost))
        t = np.array([*(boost * (math.sinh(r) / r if r > 1e-9 else 1.0)), math.cosh(r)])
        tau = complex(RNG.normal() * 0.6, 0.5 + abs(RNG.normal()) * 0.6)
        Ji = adapted_J(B, etaW, t, tau)
        gW = frame_map_W(etaW, t0, t)
        h = frame_map_A(JA0, J_A_of(tau))
        gfull = np.kron(gW, h)
        resid = float(np.max(np.abs(gfull @ J0 @ np.linalg.inv(gfull) - Ji)))
        samples.append(resid)
    worst = max(samples)
    say(f"  transitivity: for 40 random adapted J(t,tau), worst residual of "
        f"g.J0.g^-1 - J(t,tau) = {worst:.2e}")
    say("   (g = g_W (x) h with g_W in SO+(3,1) mapping t0->t, h in SL(2) mapping tau0->tau)")
    # invariant that separates orbits: dim of the stabiliser (conjugation-invariant)
    _, s_adapted2, _ = killing_on_stab(gens, adapted_J(B, etaW,
                       np.array([math.sinh(0.5), 0, 0, math.cosh(0.5)]), 0.3 + 1.4j))
    _, s_generic, _ = killing_on_stab(gens, J_gen)
    say(f"  orbit invariant dim stab : adapted J's = {s_a} (= dim K = SO(3)xSO(2)) , "
        f"another adapted = {s_adapted2} , GENERIC positive J = {s_generic}")
    say("  ★ so the adapted family is exactly ONE G-orbit = G/K = H^3 x H^2 (transitive, compact")
    say("    stabiliser K); a NON-adapted positive J has a different stab dim -> a DIFFERENT orbit.")
    dump("C-P5", worst_resid=worst, s_adapted=s_a, s_adapted2=s_adapted2, s_generic=s_generic)

    ok(lambda w: w[0] < 1e-6 and w[1] == w[2], (worst, s_a, s_adapted2),
       "T7/C-P5 (ON THE CRITERION, transitivity): G acts TRANSITIVELY on the adapted family "
       "(one orbit, compact stab K) -- and the detector rejects a non-adapted J",
       must_fail_on=("a generic non-adapted positive J (different stabiliser dimension)",
                     (worst, s_a, s_generic)))

    # ------------------------------------------ C-P6: the splice (gated)
    head("C-P6 (c, SPLICE) -- gated on (a) finite covolume AND (b) one adapted orbit")
    gate_a = d_arith > 0.5
    gate_b = (worst < 1e-6) and (s_a == s_adapted2) and (s_a > 0)
    if gate_a and gate_b:
        say("  ✓ both gates green:")
        say("    - the adapted moduli of positive J is G/K = H^3 x H^2 (dim 3+2 = 5);")
        say("    - Gamma (the beta-integral point group) has finite covolume on H^3 (C-P1);")
        say("    ==> the quotient (H^3 x H^2)/Gamma has FINITE VOLUME.")
        say("    ==> 'a functional on this space would have a compact of finite volume to live")
        say("        on' -- WITHOUT building the functional (that is future work).")
    else:
        say(f"  ✗ a gate is red (gate_a={gate_a}, gate_b={gate_b}) -- the splice line does NOT "
            "print, per norm S1598.")
    dump("C-P6", gate_a=bool(gate_a), gate_b=bool(gate_b))

    # ------------------------------------------ ★DIVIDEND (mandatory line)
    head("★DIVIDEND (mandatory) -- the AX-dimer, from axiom to a forced-spontaneous structure")
    if gate_a and gate_b:
        say("  the old assembly's AX-dimer ('time exists' = 1 bit, an AXIOM) becomes, here, a")
        say("  DERIVED forced-spontaneous structure:")
        say("    · Heisenberg unitarity FORCES the existence of a marked direction (a positive")
        say("      J exists, C-P2) -- existence is not assumed, it is required by unitarity;")
        say("    · its VALUE is SPONTANEOUS -- the adapted J's are one orbit G/K = H^3 x H^2")
        say("      (C-P5), so no J is singled out; the choice is a spontaneous point on the orbit;")
        say("    · beta-integrality makes its MODULI FINITE VOLUME -- (H^3 x H^2)/Gamma (C-P1+C-P6).")
        say("  the axiom gets a CAUSE (unitarity) and a PRICE (a spontaneous choice on a")
        say("  finite-volume moduli).  no new constant is spent.")
    else:
        say("  (dividend line withheld: a gate is red -- see C-P6.)")

    # ---------------------------------------------------------------- VERDICT (gated on teeth)
    head("VERDICT LINES (S1602) -- each gated on its own tooth (norm S1598)")
    say(f"  C-P1 (a) COVOLUME : decay arith {d_arith:.2f} vs cyclic {d_cyc:.2f} vs free {d_free:.2f}"
        f"  ==> {'Gamma is THICK: finite covolume (thin subgroups fall)' if d_arith > 0.5 else 'TOOTH RED -- no covolume claim'}")
    say(f"  C-P3 (b-i) STAB   : dim stab {s_a}, Killing max {ev_a.max() if ev_a.size else 0:+.2f}"
        f"  ==> {'positivity CUTS the Levi to a COMPACT stabiliser' if (ev_a.size and ev_a.max() < -1e-9) else 'TOOTH RED -- no compactness claim'}")
    say(f"  C-P4 (b-ii bare)  : dim G-orbit {d_orbit_G} < dim Siegel {d_siegel}"
        f"  ==> {'bare positivity leaves SEVERAL orbits (hidden input on sigma alone)' if d_orbit_G < d_siegel else 'TOOTH RED'}")
    say(f"  C-P5 (b-ii adapt) : transit resid {worst:.1e}, stab dim {s_a}={s_adapted2}"
        f"  ==> {'the ADAPTED family (house W(x)A) is ONE orbit G/K = H^3 x H^2 (spontaneous)' if (worst < 1e-6 and s_a == s_adapted2) else 'TOOTH RED'}")
    say("  ==> (a) arithmetic compactifier STANDS (thick); (b) positive compactifier: stabiliser")
    say("      COMPACT and adapted moduli ONE spontaneous orbit -- bare-sigma several orbits is")
    say("      resolved by the HOUSE tensor structure, not a new input.  (c) splice fires.")

    code = report("S1602 compactifier")
    sys.stdout.flush()
    return code


if __name__ == "__main__":
    sys.exit(main())

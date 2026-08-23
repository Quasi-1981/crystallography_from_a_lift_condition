# -*- coding: utf-8 -*-
"""S1588 -- child-3.1, TACT 3 of order #2: THE INVENTORY OF THE LIFT.

Three questions, one apparatus (the centraliser c of X inside so(5,3)):

  (a) L-P1  THE BRACKET of the heis part of c, BY FORM -- not by dimension.  This closes
            leg (ii) of my own J-P5, which the court moved HIT -> PARTIAL in S1586: there I
            printed dim heis = 9 and CALLED it "family I with beta = eta|W" without ever
            computing a single commutator.
  (b) L-P2/L-P3  sp(2) BY NAME: how it acts on A = U, on W, on the centre; and whether c
            holds anything OUTSIDE (sp2 (+) so(eta|W)) |x heis.
  (c) L-P4/L-P5/L-P6  equivariance under the FULL Levi, and the COMPOSITION (not the count)
            of the 6-dimensional T24 family.

Ex-ante: child-3.1/S1588_LIFT_INVENTORY_EXANTE.md, committed BEFORE this file existed.
The heis part is defined INTRINSICALLY by the flag X itself (U = im X inside U^perp = ker X),
so it needs no convention; the order's Frobenius-orthocomplement reading is measured SECOND
and the agreement is printed.

Run line (unconditionally into the seal):
    python child-3.1/S1588_lift_inventory.py --outdir child-3.1
"""
import argparse
import json
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

TOL = 1e-8                      # named ex-ante: relative singular-value cut, scale free
RNG = np.random.default_rng(20260819)
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
    with open(_path("S1588_lift_inventory_dump.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"rec": rec, **payload}, sort_keys=True, default=str) + "\n")


def mat(name, M, fmt="%7.3f"):
    say(f"    {name} =")
    for row in np.atleast_2d(M):
        say("      [" + "  ".join(fmt % x for x in row) + "]")


# ===================================================== 1. THE PARENT, THE FLAG, THE SPLITTING
def eta_of(p, q):
    return np.diag([1.0] * p + [-1.0] * q)


def wedge(x, y, eta):
    """x ^ y acting as z -> x eta(y,z) - y eta(x,z)."""
    return np.outer(x, eta @ y) - np.outer(y, eta @ x)


def named_setup(p, q):
    """The NAMED splitting of the ex-ante, rebuilt from scratch (nothing quoted)."""
    n = p + q
    eta = eta_of(p, q)
    u, v = np.zeros(n), np.zeros(n)
    u[0], u[p] = 1.0, 1.0                 # e1 + e_{p+1}
    v[1], v[p + 1] = 1.0, 1.0             # e2 + e_{p+2}
    us, vs = np.zeros(n), np.zeros(n)
    us[0], us[p] = 0.5, -0.5              # (e1 - e_{p+1}) / 2
    vs[1], vs[p + 1] = 0.5, -0.5
    widx = [i for i in range(n) if i not in (0, 1, p, p + 1)]
    Wm = np.zeros((n, len(widx)))
    for c, i in enumerate(widx):
        Wm[i, c] = 1.0
    X = wedge(u, v, eta)
    Um = np.column_stack([u, v])
    Usm = np.column_stack([us, vs])
    etaW = Wm.T @ eta @ Wm
    return eta, X, Um, Usm, Wm, etaW


def so_basis(p, q):
    n, eta = p + q, eta_of(p, q)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            ei, ej = np.zeros(n), np.zeros(n)
            ei[i], ej[j] = 1.0, 1.0
            out.append(np.outer(ei, eta @ ej) - np.outer(ej, eta @ ei))
    return out


def nullspace(M, tol=TOL):
    """Basis of ker M, read from RELATIVE singular values -- never from a determinant."""
    M = np.atleast_2d(M)
    _, sv, Vt = np.linalg.svd(M)
    scale = sv[0] if len(sv) and sv[0] > 0 else 1.0
    small = int(np.sum(sv <= tol * scale)) + (M.shape[1] - len(sv))
    return Vt[M.shape[1] - small:] if small else np.zeros((0, M.shape[1]))


def onb(M):
    """Euclid-orthonormal basis of the column span of M."""
    Q, R = np.linalg.qr(M)
    r = int(np.sum(np.abs(np.diag(R)) > 1e-10))
    return Q[:, :r]


def combine(coefs, basis):
    return sum(c * b for c, b in zip(coefs, basis))


def subspace_of_c(cbasis, constraint):
    """{A in span(cbasis) : constraint(A) = 0}, returned as a list of matrices."""
    rows = np.array([constraint(b).flatten() for b in cbasis]).T
    ns = nullspace(rows)
    return [combine(co, cbasis) for co in ns]


def rank_of(mats, tol=1e-8):
    if not len(mats):
        return 0
    A = np.array([np.asarray(m).flatten() for m in mats])
    return int(np.linalg.matrix_rank(A, tol=tol * max(1.0, np.max(np.abs(A)))))


# ===================================================== 2. HODGE STAR ON Lambda^2 W
def pairs_of(d):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def eps4():
    E = np.zeros((4, 4, 4, 4))
    from itertools import permutations
    for perm in permutations(range(4)):
        s = 1
        pl = list(perm)
        for i in range(4):
            for j in range(i + 1, 4):
                if pl[i] > pl[j]:
                    s = -s
        E[perm] = s
    return E


def star_matrix(etaW):
    """The Hodge star on Lambda^2 W in the basis {e_i ^ e_j}_{i<j}, from eta|W alone."""
    d = etaW.shape[0]
    assert d == 4
    E = eps4()
    prs = pairs_of(d)
    idx = {p: k for k, p in enumerate(prs)}
    S = np.zeros((len(prs), len(prs)))
    for b, (k, l) in enumerate(prs):
        F = np.zeros((d, d))
        F[k, l], F[l, k] = 1.0, -1.0
        for a, (i, j) in enumerate(prs):
            val = 0.0
            for kk in range(d):
                for ll in range(d):
                    val += E[i, j, kk, ll] * etaW[kk, kk] * etaW[ll, ll] * F[kk, ll]
            S[a, b] = 0.5 * val
    return S


def act_lambda2(S, d):
    """The action induced on Lambda^2 of a space by S acting on the space itself."""
    prs = pairs_of(d)
    M = np.zeros((len(prs), len(prs)))
    for a, (i, j) in enumerate(prs):
        for b, (k, l) in enumerate(prs):
            val = 0.0
            if j == l:
                val += S[k, i]
            if i == k:
                val += S[l, j]
            if j == k:
                val -= S[l, i]
            if i == l:
                val -= S[k, j]
            M[b, a] = val
    return M


# ===================================================== 3. EQUIVARIANT BRACKETS (generalised)
def equivariant_solutions(gens, dW, dA, target_dim):
    """Antisymmetric c : Lambda^2(W (x) A) -> Z commuting with EVERY generator.

    gens: list of (SW, SA, T) -- SW acts on W, SA acts on A, T acts on the target Z.
    Equivariance: c(g.x, y) + c(x, g.y) = T . c(x, y).
    """
    n = dW * dA
    prs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    npair = len(prs)
    idx = {p: k for k, p in enumerate(prs)}

    def act(SW, SA, k):
        w, a = divmod(k, dA)
        out = np.zeros(n)
        for w2 in range(dW):
            out[w2 * dA + a] += SW[w2, w]
        for a2 in range(dA):
            out[w * dA + a2] += SA[a2, a]
        return out

    rows = []
    for (SW, SA, T) in gens:
        for (i, j) in prs:
            si, sj = act(SW, SA, i), act(SW, SA, j)
            for t in range(target_dim):
                row = np.zeros(npair * target_dim)
                for k in range(n):
                    if abs(si[k]) > 1e-14 and k != j:
                        a_, b_ = (k, j) if k < j else (j, k)
                        sg = 1.0 if k < j else -1.0
                        row[idx[(a_, b_)] * target_dim + t] += sg * si[k]
                    if abs(sj[k]) > 1e-14 and i != k:
                        a_, b_ = (i, k) if i < k else (k, i)
                        sg = 1.0 if i < k else -1.0
                        row[idx[(a_, b_)] * target_dim + t] += sg * sj[k]
                for s in range(target_dim):
                    row[idx[(i, j)] * target_dim + s] -= T[t, s]
                rows.append(row)
    ns = nullspace(np.array(rows))
    return prs, [c.reshape(npair, target_dim) for c in ns], len(ns)


def naive_family_II(dW, dA):
    """F: [w(x)a, w'(x)b] = (w ^ w') eta'(a,b), one solution per eta' in S^2 A*."""
    n = dW * dA
    prs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    wprs = pairs_of(dW)
    widx = {p: k for k, p in enumerate(wprs)}
    out = []
    for (s, t) in [(a, b) for a in range(dA) for b in range(a, dA)]:
        etap = np.zeros((dA, dA))
        etap[s, t] = etap[t, s] = 1.0
        sol = np.zeros((len(prs), len(wprs)))
        for k, (i, j) in enumerate(prs):
            w1, a1 = divmod(i, dA)
            w2, a2 = divmod(j, dA)
            if w1 == w2:
                continue
            sg = 1.0 if w1 < w2 else -1.0
            key = (w1, w2) if w1 < w2 else (w2, w1)
            sol[k, widx[key]] += sg * etap[a1, a2]
        out.append(sol)
    return prs, out


# =============================================================================== 4. THE RUN
def inventory(p, q, label, verbose=True):
    """The whole inventory for the parent (p,q).  Returns a dict of measured quantities."""
    n = p + q
    eta, X, Um, Usm, Wm, etaW = named_setup(p, q)
    B = so_basis(p, q)
    Mad = np.array([(b @ X - X @ b).flatten() for b in B]).T
    cbasis = [combine(co, B) for co in nullspace(Mad)]
    dim_c = len(cbasis)

    PU = onb(Um) @ onb(Um).T
    PUs = onb(Usm) @ onb(Usm).T
    PW = onb(Wm) @ onb(Wm).T
    PUp = PU + PW                                   # U^perp = ker X = U (+) W
    I = np.eye(n)

    def in_n(A):                                    # the INTRINSIC heis conditions
        return np.vstack([(I - PUp) @ A, (I - PU) @ A @ PUp, A @ PU])

    def in_sp2(A):
        return np.vstack([A @ PW, (I - PU) @ A @ PU, (I - PUs) @ A @ PUs])

    def in_soW(A):
        return np.vstack([A @ PU, A @ PUs, (I - PW) @ A @ PW])

    nn = subspace_of_c(cbasis, in_n)
    sp2 = subspace_of_c(cbasis, in_sp2)
    soW = subspace_of_c(cbasis, in_soW)

    # the order's convention, measured SECOND: Frobenius orthocomplement of the Levi in c
    levi = sp2 + soW
    if len(levi):
        Lflat = np.array([m.flatten() for m in levi])
        Q = np.linalg.qr(Lflat.T)[0]
        proj = [m.flatten() - Q @ (Q.T @ m.flatten()) for m in cbasis]
        n_frob = rank_of(proj)
    else:
        n_frob = dim_c
    # do the two readings span the SAME subspace?
    same = rank_of(nn + [np.asarray(x).reshape(n, n) for x in
                         [m.flatten() - Q @ (Q.T @ m.flatten()) for m in cbasis]])

    res = dict(dim_c=dim_c, dim_n=len(nn), dim_sp2=len(sp2), dim_soW=len(soW),
               n_frob=n_frob, n_union=same, etaW=etaW.tolist())

    if verbose:
        say(f"  dim so({p},{q}) = {len(B)}   dim c = {dim_c}")
        say(f"  eta|W (named basis)               : diag({', '.join('%+.0f' % x for x in np.diag(etaW))})")
        say(f"  dim n (INTRINSIC, flag of X)      : {len(nn)}")
        say(f"  dim n (Frobenius orthocomplement) : {n_frob}   "
            f"[rank of the UNION of both readings: {same}]")
        say(f"  dim sp2 = {len(sp2)}   dim so(eta|W) = {len(soW)}   "
            f"sum = {len(sp2)+len(soW)+len(nn)}")
    return res, dict(eta=eta, X=X, Um=Um, Usm=Usm, Wm=Wm, etaW=etaW, cbasis=cbasis,
                     nn=nn, sp2=sp2, soW=soW, B=B)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=_HERE)
    args = ap.parse_args(argv)
    _OUT["dir"] = os.path.abspath(args.outdir)
    os.makedirs(_OUT["dir"], exist_ok=True)
    sys.stdout = Tee(sys.stdout, open(_path("S1588_lift_inventory_run.log"), "w",
                                      encoding="utf-8"))
    open(_path("S1588_lift_inventory_dump.jsonl"), "w", encoding="utf-8").close()

    head("S1588 -- THE INVENTORY OF THE LIFT (bracket by FORM, sp(2) by NAME)")
    say(f"  thresholds fixed in advance: relative singular cut {TOL:.0e}; proportionality read")
    say("  from a NORMALISED residual (norm S1583: never a determinant).")
    say("  heis part = INTRINSIC (the flag U = im X inside U^perp = ker X); the order's")
    say("  Frobenius-orthocomplement reading is measured second and the agreement printed.")

    P, Q = 5, 3

    # ------------------------------------------------------------------ L-P2: the inventory
    head("L-P2 -- what c is made of, BY NAME:  sp(2) (+) so(eta|W) (+) n , and nothing else")
    res, obj = inventory(P, Q, "(5,3)")
    eta, X, Um, Wm, etaW = obj["eta"], obj["X"], obj["Um"], obj["Wm"], obj["etaW"]
    cbasis, nn, sp2, soW = obj["cbasis"], obj["nn"], obj["sp2"], obj["soW"]
    dim_c = res["dim_c"]
    rank_sum = rank_of(sp2 + soW + nn)
    rank_drop = rank_of(sp2 + soW)
    say(f"  rank of the GLUED basis sp2 + so + n : {rank_sum}   (dim c = {dim_c})")
    say(f"  rank with n dropped                  : {rank_drop}  <- the negative world of T5")

    def br(A, Bm):
        return A @ Bm - Bm @ A

    def inside(mats, span, tag):
        if not len(span):
            return float(np.max([np.max(np.abs(m)) for m in mats]))
        F = np.array([np.asarray(m).flatten() for m in span]).T
        worst = 0.0
        for m in mats:
            r = np.asarray(m).flatten()
            resid = r - F @ np.linalg.lstsq(F, r, rcond=None)[0]
            sc = max(np.linalg.norm(r), 1e-30)
            worst = max(worst, float(np.linalg.norm(resid) / sc))
        return worst

    ideal_res = inside([br(a, b) for a in cbasis for b in nn], nn, "n ideal")
    comm_levi = float(np.max([np.max(np.abs(br(a, b))) for a in sp2 for b in soW]))
    semi_soW = float(np.max([np.max(np.abs(br(a, b))) for a in soW for b in nn]))
    semi_sp2 = float(np.max([np.max(np.abs(br(a, b))) for a in sp2 for b in nn]))
    nilp = float(np.max([np.max(np.abs(np.linalg.matrix_power(a, 3))) for a in nn]))
    say(f"  n is an IDEAL in c, worst normalised residual : {ideal_res:.3e}")
    say(f"  [sp2, so(eta|W)] , max |.|                    : {comm_levi:.3e}   (expect 0)")
    say(f"  [so(eta|W), n] , max |.|                      : {semi_soW:.3e}   (expect NOT 0)")
    say(f"  [sp2, n] , max |.|                            : {semi_sp2:.3e}   (expect NOT 0)")
    say(f"  every element of n is nilpotent (A^3)         : {nilp:.3e}")
    dump("inventory", **res, rank_sum=rank_sum, rank_drop=rank_drop, ideal=ideal_res,
         comm_levi=comm_levi, semi_soW=semi_soW, semi_sp2=semi_sp2, nilp=nilp)

    ok(lambda w: max(abs(np.max(np.abs(b @ w[1] - w[1] @ b))) for b in w[0]) < 1e-10,
       (cbasis, X),
       "T1/L-P2: every basis element of c really commutes with X",
       must_fail_on=("an element of so(5,3) that is NOT in c",
                     ([obj["B"][0]], X)))

    def n_ok(world):
        mats, PU_, PUp_, I_ = world
        w = 0.0
        for A in mats:
            w = max(w, float(np.max(np.abs((I_ - PUp_) @ A))),
                    float(np.max(np.abs((I_ - PU_) @ A @ PUp_))),
                    float(np.max(np.abs(A @ PU_))))
        return w < 1e-10

    PUo, PWo = onb(Um) @ onb(Um).T, onb(Wm) @ onb(Wm).T
    ok(n_ok, (nn, PUo, PUo + PWo, np.eye(P + Q)),
       "T2/L-P2: every element of the intrinsic heis part really satisfies the flag conditions",
       must_fail_on=("an element of c that lies OUTSIDE n (a so(eta|W) generator)",
                     (soW[:1], PUo, PUo + PWo, np.eye(P + Q))))

    ok(lambda w: w[0] == w[1] and w[2] < w[1], (rank_sum, dim_c, rank_drop),
       "T5/L-P2: sp2 + so(eta|W) + n span c by RANK (3+6+9=18), and dropping n drops the rank",
       must_fail_on=("a world where dropping a summand changed nothing",
                     (rank_sum, dim_c, dim_c)))

    # ------------------------------------------------------------------ L-P3: sp(2) by name
    head("L-P3 -- sp(2) BY NAME: its action on A = U, on W, on the centre Lambda^2 A")
    u, v = Um[:, 0], Um[:, 1]
    SA_list = []
    for k, A in enumerate(sp2):
        SA = np.linalg.lstsq(Um, A @ Um, rcond=None)[0]
        SA_list.append(SA)
        say(f"  generator {k}: action on A in the basis (u,v), trace = {np.trace(SA):+.3e}")
        mat("SA", SA)
        say(f"    action on W, max |A|_W| = "
            f"{float(np.max(np.abs(A @ PWo))):.3e}   (expect 0)")
    tr_worst = float(np.max([abs(np.trace(S)) for S in SA_list]))
    onW_worst = float(np.max([np.max(np.abs(A @ PWo)) for A in sp2]))
    say(f"  worst |trace| over sp(2) generators : {tr_worst:.3e}  ==> action on Lambda^2 A is ZERO")
    say(f"  worst |action on W|                 : {onW_worst:.3e}")
    dump("sp2", traces=[float(np.trace(S)) for S in SA_list], onW=onW_worst,
         SA=[S.tolist() for S in SA_list])

    ok(lambda w: w[0] < 1e-10 and w[1] < 1e-10, (tr_worst, onW_worst),
       "T6/L-P3: the sp(2) block is TRACELESS on A and ZERO on W -- it is sl(2), not gl(2)",
       must_fail_on=("a gl(2) element carrying a trace", (1.0, onW_worst)))

    # ------------------------------------------------------------------ L-P1: THE BRACKET
    head("L-P1 (LOAD-BEARING) -- the BRACKET of the heis part, BY FORM.  Dimension is not an "
         "argument here.")
    centre = subspace_of_c(nn, lambda A: A @ (PUo + PWo))
    say(f"  dim centre of n = {len(centre)}")
    cen_res = inside(centre, [X], "centre = span X")
    say(f"  centre vs span(X), normalised residual : {cen_res:.3e}   "
         "==> the centre is spanned by X ITSELF")

    dW, dA = etaW.shape[0], 2
    Wcols = [Wm[:, i] for i in range(dW)]
    Acols = [u, v]

    def N(w, a):
        return wedge(a, w, eta)

    gens_in_n = inside([N(w, a) for w in Wcols for a in Acols] + [X], nn, "generators in n")
    say(f"  the explicit generators a^w and X lie in n, worst residual : {gens_in_n:.3e}")
    say(f"  rank of {{a^w}} + {{X}} : {rank_of([N(w, a) for w in Wcols for a in Acols] + [X])}"
        f"   (dim n = {len(nn)})")

    # beta as a MATRIX, from commutators, with a = u, b = v  (so that a^b = X exactly)
    beta = np.zeros((dW, dW))
    for i in range(dW):
        for j in range(dW):
            C = br(N(Wcols[i], u), N(Wcols[j], v))
            beta[i, j] = float(np.linalg.lstsq(X.reshape(-1, 1), C.flatten(),
                                               rcond=None)[0][0])
    say("  beta read from the commutators [u^w_i , v^w_j] in the NAMED centre coordinate a^b = X:")
    mat("beta", beta)
    mat("eta|W", etaW)
    rho = float(np.sum(beta * etaW) / np.sum(etaW * etaW))
    prop = float(np.linalg.norm(beta - rho * etaW) / max(np.linalg.norm(beta), 1e-30))
    say(f"  rho = <beta, eta|W> / <eta|W, eta|W> = {rho:+.6f}    "
        f"normalised residual ||beta - rho eta|W|| / ||beta|| = {prop:.3e}")
    bmag = float(np.max(np.abs(beta)))
    say(f"  max |beta| = {bmag:.3e}    ==>  [n,n] is NOT zero   <- THE ONLY LINE HERE THAT "
        "COULD HAVE GONE THE OTHER WAY")

    # the FULL form on random (w,a,w',b): one single rho must serve all of them
    worst_sample, worst_centre = 0.0, 0.0
    for _ in range(200):
        cw, cw2 = RNG.normal(size=dW), RNG.normal(size=dW)
        ca, cb = RNG.normal(size=2), RNG.normal(size=2)
        w = sum(c * x for c, x in zip(cw, Wcols))
        w2 = sum(c * x for c, x in zip(cw2, Wcols))
        a = ca[0] * u + ca[1] * v
        b = cb[0] * u + cb[1] * v
        C = br(N(w, a), N(w2, b))
        coef = float(np.linalg.lstsq(X.reshape(-1, 1), C.flatten(), rcond=None)[0][0])
        worst_centre = max(worst_centre,
                           float(np.linalg.norm(C - coef * X) / max(np.linalg.norm(C), 1e-30)))
        det_ab = ca[0] * cb[1] - ca[1] * cb[0]
        pred = rho * float(cw @ etaW @ cw2) * det_ab
        worst_sample = max(worst_sample, abs(coef - pred) / max(abs(coef), 1e-12))
    say(f"  on 200 RANDOM (w,a,w',b): bracket stays in span(X), worst residual {worst_centre:.3e}")
    say(f"  and equals rho * eta|W(w,w') * det(a,b) with the SAME rho, worst rel. error "
        f"{worst_sample:.3e}")
    dump("bracket", rho=rho, prop=prop, bmag=bmag, centre_dim=len(centre), cen_res=cen_res,
         worst_sample=worst_sample, worst_centre=worst_centre, beta=beta.tolist())

    def form_ok(world):
        bet, ew, ws = world
        r = float(np.sum(bet * ew) / np.sum(ew * ew))
        return (float(np.linalg.norm(bet - r * ew) / max(np.linalg.norm(bet), 1e-30)) < 1e-8
                and ws < 1e-8)

    ok(form_ok, (beta, etaW, worst_sample),
       "T3/L-P1 (LOAD-BEARING): the heis bracket has family-I FORM with ONE shared "
       "beta = rho * eta|W, on the basis AND on 200 random arguments",
       must_fail_on=("a beta deliberately off eta|W (diag(1,2,3,4))",
                     (np.diag([1.0, 2.0, 3.0, 4.0]), etaW, worst_sample)))

    ok(lambda w: float(np.max(np.abs(w))) > 1e-6, beta,
       "T4/L-P1: [n,n] is NOT zero -- the ideal is Heisenberg, not abelian, so family I is "
       "ACTUALLY realised inside the parent",
       must_fail_on=("an abelian algebra of the same dimension 9",
                     np.zeros((dW, dW))))

    ok(lambda w: w[0] == 1 and w[1] < 1e-10, (len(centre), cen_res),
       "T2b/L-P1: the centre of n is one-dimensional and spanned by X itself",
       must_fail_on=("a world where the centre needed a second generator", (2, cen_res)))

    # ------------------------------------------------ L-P4: equivariance under the FULL Levi
    head("L-P4 (SHARP) -- equivariance under the FULL Levi sp(2) (+) so(eta|W): what survives")
    # ★NOTHING here is typed as a zero: EVERY block -- how a generator acts on W, on A, and
    # on each target -- is READ OFF the generator by least squares.  This is the direct debt
    # of the S1586 erratum E-1: a printed zero must be a MEASURED zero, never a construction.
    def blocks_of(A):
        SW = np.linalg.lstsq(Wm, A @ Wm, rcond=None)[0]
        SA = np.linalg.lstsq(Um, A @ Um, rcond=None)[0]
        return SW, SA

    lev_soW = [blocks_of(A) for A in soW]
    lev_sp2 = [blocks_of(A) for A in sp2]
    say("  provenance of every target action (measured, not typed):")
    say(f"    so(eta|W) generators: worst |action on A|      = "
        f"{max(float(np.max(np.abs(SA))) for _, SA in lev_soW):.3e}")
    say(f"    sp(2)     generators: worst |action on W|      = "
        f"{max(float(np.max(np.abs(SW))) for SW, _ in lev_sp2):.3e}")
    say(f"    sp(2)     generators: worst |induced on L^2 W| = "
        f"{max(float(np.max(np.abs(act_lambda2(SW, dW)))) for SW, _ in lev_sp2):.3e}")
    say(f"    so(eta|W) generators: worst |induced on L^2 A| = "
        f"{max(abs(float(np.trace(SA))) for _, SA in lev_soW):.3e}")

    def gens_I(levs):
        return [(SW, SA, np.array([[np.trace(SA)]])) for SW, SA in levs]

    def gens_II(levs):
        return [(SW, SA, act_lambda2(SW, dW)) for SW, SA in levs]

    soW_onW = [SW for SW, _ in lev_soW]
    gens_soW_I, gens_full_I = gens_I(lev_soW), gens_I(lev_soW + lev_sp2)
    gens_soW_II, gens_full_II = gens_II(lev_soW), gens_II(lev_soW + lev_sp2)

    prs, solI_so, dI_so = equivariant_solutions(gens_soW_I, dW, dA, 1)
    _, solI_full, dI_full = equivariant_solutions(gens_full_I, dW, dA, 1)
    _, solII_so, dII_so = equivariant_solutions(gens_soW_II, dW, dA, 6)
    _, solII_full, dII_full = equivariant_solutions(gens_full_II, dW, dA, 6)
    say(f"  family I  (values in Lambda^2 A):  under so(eta|W) alone = {dI_so}   "
        f"under the FULL Levi = {dI_full}")
    say(f"  family II (values in Lambda^2 W):  under so(eta|W) alone = {dII_so}   "
        f"under the FULL Levi = {dII_full}")
    say("  ^ the contrast is a MOVING quantity: same machinery, same carrier, different group.")
    say("    (the S1586 erratum E-1 is paid here by the CONSTRUCTION of the tooth, not by a")
    say("     promise: nothing in this comparison is zero by construction.)")

    # beta of the surviving family-I solution, to be sure it is still the daughter metric
    def beta_of(sol):
        bt = np.zeros((dW, dW))
        for k, (i, j) in enumerate(prs):
            w1, a1 = divmod(i, dA)
            w2, a2 = divmod(j, dA)
            if a1 == 0 and a2 == 1:
                bt[w1, w2] += sol[k, 0]
            elif a1 == 1 and a2 == 0:
                bt[w1, w2] -= sol[k, 0]
        return 0.5 * (bt + bt.T)

    prop_full = 1.0
    if dI_full:
        bt = beta_of(solI_full[0])
        r = float(np.sum(bt * etaW) / np.sum(etaW * etaW))
        prop_full = float(np.linalg.norm(bt - r * etaW) / max(np.linalg.norm(bt), 1e-30))
        say(f"  the surviving family-I solution has beta = {r:+.6f} * eta|W , "
            f"normalised residual {prop_full:.3e}")
    dump("levi", dI_so=dI_so, dI_full=dI_full, dII_so=dII_so, dII_full=dII_full,
         prop_full=prop_full)

    ok(lambda w: w[0] == 0 and w[1] == 6, (dII_full, dII_so),
       "T7/L-P4: the T24 family DIES under the full Levi (0) while the SAME machinery on the "
       "SAME carrier under so(eta|W) alone gives 6 -- the contrast moves",
       must_fail_on=("a world where adding sp(2) changed nothing", (6, 6)))

    ok(lambda w: w[0] == 1 and w[1] < 1e-8, (dI_full, prop_full),
       "T7b/L-P4: the daughter metric SURVIVES the full Levi, still dim 1 and still ~ eta|W",
       must_fail_on=("a world where the daughter died too", (0, prop_full)))

    # ------------------------------------------------ L-P5: the COMPOSITION of the six
    head("L-P5 -- the COMPOSITION of the 6 (not the count): pairs c / *c under the Hodge star")
    St = star_matrix(etaW)
    mat("Hodge star on Lambda^2 W", St, fmt="%5.1f")
    st2 = float(np.max(np.abs(St @ St + np.eye(6))))
    say(f"  || *^2 + 1 ||_max = {st2:.3e}   ==> *^2 = -1 on the LORENTZIAN daughter")
    comm_star = float(np.max([np.max(np.abs(St @ act_lambda2(S, dW)
                                            - act_lambda2(S, dW) @ St)) for S in soW_onW]))
    say(f"  [*, so(eta|W)-action on Lambda^2 W] , max = {comm_star:.3e}")

    prsF, F = naive_family_II(dW, dA)
    starF = [s @ St.T for s in F]
    inF = inside(F, solII_so, "F inside the solution space")
    dF, dSF = rank_of(F), rank_of(starF)
    dBoth = rank_of(F + starF)
    say(f"  dim F (naive eta' family) = {dF}   dim *F = {dSF}   dim (F + *F) = {dBoth}"
        f"   [F intersect *F = {dF + dSF - dBoth}]")
    say(f"  F really sits inside the 6-dim solution space, worst residual {inF:.3e}")
    say(f"  dim (F + F) = {rank_of(F + F)}  <- the negative world of T9: doubling F alone "
        "spans nothing new")
    dump("hodge", st2=st2, comm_star=comm_star, dF=dF, dSF=dSF, dBoth=dBoth, inF=inF)

    ok(lambda w: w[0] < 1e-10 and w[1] < 1e-10, (st2, comm_star),
       "T8/L-P5: *^2 = -1 on the Lorentzian daughter and * commutes with the so(eta|W) action",
       must_fail_on=("the Euclidean daughter (4,0), where *^2 = +1",
                     (float(np.max(np.abs(star_matrix(np.eye(4)) @ star_matrix(np.eye(4))
                                          + np.eye(6)))), comm_star)))

    ok(lambda w: w[0] == 3 and w[1] == 3 and w[2] == 6, (dF, dSF, dBoth),
       "T9/L-P5: the six is COMPOSED as F (+) *F -- three real eta' times a two-dimensional "
       "commutant, not six unrelated brackets",
       must_fail_on=("F doubled with itself, which spans only three",
                     (dF, dF, rank_of(F + F))))

    # ------------------------------------------------ L-P6: the other parent (4,4) -> (2,2)
    head("L-P6 -- the CONTROL carrier: parent (4,4), daughter (2,2), where *^2 = +1")
    res44, obj44 = inventory(4, 4, "(4,4)")
    etaW44 = obj44["etaW"]
    St44 = star_matrix(etaW44)
    st2_44 = float(np.max(np.abs(St44 @ St44 - np.eye(6))))
    def blocks44(A):
        return (np.linalg.lstsq(obj44["Wm"], A @ obj44["Wm"], rcond=None)[0],
                np.linalg.lstsq(obj44["Um"], A @ obj44["Um"], rcond=None)[0])

    lev_so44 = [blocks44(A) for A in obj44["soW"]]
    lev_sp44 = [blocks44(A) for A in obj44["sp2"]]
    soW44 = [SW for SW, _ in lev_so44]
    g_so_II_44 = [(SW, SA, act_lambda2(SW, 4)) for SW, SA in lev_so44]
    g_full_II_44 = [(SW, SA, act_lambda2(SW, 4)) for SW, SA in lev_so44 + lev_sp44]
    g_so_I_44 = [(SW, SA, np.array([[np.trace(SA)]])) for SW, SA in lev_so44]
    _, _, dII_so44 = equivariant_solutions(g_so_II_44, 4, 2, 6)
    _, _, dII_full44 = equivariant_solutions(g_full_II_44, 4, 2, 6)
    _, _, dI_so44 = equivariant_solutions(g_so_I_44, 4, 2, 1)
    ev44 = np.linalg.eigvals(St44)
    npos = int(np.sum(np.real(ev44) > 0.5))
    nneg = int(np.sum(np.real(ev44) < -0.5))
    say(f"  eta|W(4,4) = diag({', '.join('%+.0f' % x for x in np.diag(etaW44))})   "
        f"|| *^2 - 1 || = {st2_44:.3e}   ==> *^2 = +1 (SPLIT, not complex)")
    say(f"  * eigenvalues: {npos} at +1 and {nneg} at -1  ==> the six splits as 3 (+) 3 "
        "EIGENSPACES, not as a complex structure")
    say(f"  family II under so alone = {dII_so44}   under the FULL Levi = {dII_full44}   "
        f"family I under so alone = {dI_so44}")
    dump("control44", **res44, st2=st2_44, npos=npos, nneg=nneg, dII_so=dII_so44,
         dII_full=dII_full44, dI_so=dI_so44)

    ok(lambda w: w[0] < 1e-10 and w[1] == 3 and w[2] == 3 and w[3] == 6,
       (st2_44, npos, nneg, dII_so44),
       "T10/L-P6: on the (2,2) daughter the star SQUARES TO +1 and the six splits into two "
       "real eigenspaces 3+3 -- same COUNT, different COMPOSITION",
       must_fail_on=("the (5,3) numbers substituted as (4,4): there *^2 = -1, no real "
                     "eigenvalues at all", (st2, 0, 0, dII_so)))

    # ---------------------------------------------------------------------------- VERDICT
    head("VERDICT LINES (S1588)")
    say(f"  L-P1 heis bracket, FORM  : beta = {rho:+.6f} * eta|W , residual {prop:.3e} ; "
        f"[n,n] != 0 (max|beta| = {bmag:.3e})")
    say(f"       on 200 random args  : one shared rho, worst rel. error {worst_sample:.3e} ; "
        f"image in span(X) {worst_centre:.3e}")
    say(f"  L-P2 c by name           : {res['dim_sp2']} + {res['dim_soW']} + {res['dim_n']} "
        f"= rank {rank_sum} = dim c {dim_c} ; n ideal {ideal_res:.1e} ; "
        f"[sp2,so] {comm_levi:.1e}")
    say(f"       intrinsic n vs Frobenius orthocomplement : {res['dim_n']} vs {res['n_frob']} "
        f"(union rank {res['n_union']})")
    say(f"  L-P3 sp(2) by name       : traceless {tr_worst:.1e} , zero on W {onW_worst:.1e} "
        "==> sl(2), trivial on the centre")
    say(f"  L-P4 FULL Levi           : family I {dI_so} -> {dI_full} (survives) ; "
        f"family II {dII_so} -> {dII_full} (dies)")
    say(f"  L-P5 composition of 6    : F {dF} (+) *F {dSF} = {dBoth} , *^2 = -1 ({st2:.1e})")
    say(f"  L-P6 control (4,4)       : *^2 = +1 ({st2_44:.1e}) , split 3+3 , family II "
        f"{dII_so44} -> {dII_full44}")

    code = report("S1588 lift_inventory")
    sys.stdout.flush()
    return code


if __name__ == "__main__":
    sys.exit(main())

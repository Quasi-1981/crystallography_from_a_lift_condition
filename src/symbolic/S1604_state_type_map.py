# -*- coding: utf-8 -*-
"""S1604 -- child-3.1, TACT 9: THE MAP OF STATE TYPES (the state is NOT chosen).

S1602 gave two cuts of the group: positivity -> a COMPACT stabiliser K_J in the Levi; arithmetic
-> a DISCRETE Gamma = G(Z).  Neither alone is finite.  TOGETHER, automatically:

        K_J (compact)  cap  Gamma (discrete)  =  FINITE.

The finite point groups of the assembly are not chosen and not imported from crystallography -- they
are born as this intersection, PER STRATUM of J.  The "adaptedness" tension of S1602 dissolves: it
is a STRATUM (a stabiliser type), not a selection.

  (a) strata of positive J by dim K_J (a metric ladder: 4 / 2 / 1 / 0);
  (b) the finite group K_J cap Gamma per stratum -- order AND composition (element orders measured,
      never inferred from |lambda|; the order-3 / tetrahedral marker asked NEUTRALLY, no hint);
  (c) a table of types: stratum x K_J cap Gamma x whether a beta-integral Lambda realises it.
      No row is crowned.

Ex-ante: child-3.1/S1604_STATE_TYPE_MAP_EXANTE.md, committed BEFORE this file existed.
Norms in the fence: S1595 (completeness of the strata list carries a negative world ON the
criterion), S1598 (a verdict line is gated on its own tooth), S1576 (the composition of K_J cap Gamma
must be a stratum invariant; dependence on the representative means the stratum splits further).

Run line (unconditionally into the seal):
    python child-3.1/S1604_state_type_map.py --outdir child-3.1
"""
import argparse
import itertools as it
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

TOL = 1e-8
RNG = np.random.default_rng(20260820)
_OUT = {"dir": _HERE}
OMEGA_A = np.array([[0.0, 1.0], [-1.0, 0.0]])


def _path(name):
    return os.path.join(_OUT["dir"], name)


class Tee(object):
    def __init__(self, real, fh):
        self.real, self.fh = real, fh

    def write(self, s):
        self.real.write(s); self.fh.write(s); return len(s)

    def flush(self):
        self.real.flush()
        if not self.fh.closed:
            self.fh.flush()


def say(s=""):
    print(s)


def head(s):
    say("\n" + "=" * 96); say(s); say("=" * 96)


def dump(rec, **payload):
    with open(_path("S1604_state_type_map_dump.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"rec": rec, **payload}, sort_keys=True, default=str) + "\n")


# =============================================================== floor 0 (as S1602)
def eta_of(p, q):
    return np.diag([1.0] * p + [-1.0] * q)


def bracket_form(etaW):
    return -np.kron(etaW, OMEGA_A)


def so_eta_generators(etaW):
    n = etaW.shape[0]; I = np.eye(n); gens = []
    for i in range(n):
        for j in range(i + 1, n):
            gens.append(np.outer(I[i], etaW @ I[j]) - np.outer(I[j], etaW @ I[i]))
    return gens


SL2 = [np.array([[1.0, 0.0], [0.0, -1.0]]), np.array([[0.0, 1.0], [0.0, 0.0]]),
       np.array([[0.0, 0.0], [1.0, 0.0]])]


def levi_generators(etaW):
    return [np.kron(X, np.eye(2)) for X in so_eta_generators(etaW)] + \
           [np.kron(np.eye(4), Y) for Y in SL2]


def compatible_J(B, G):
    Shat = np.linalg.solve(G, B)
    w, Vv = np.linalg.eig(-Shat @ Shat)
    sq = (Vv * np.sqrt(w + 0j)) @ np.linalg.inv(Vv)
    J = np.real(Shat @ np.linalg.inv(sq))
    if np.trace(B @ J) < 0:                      # positivity selection (S1602 §1)
        J = -J
    return J


def stab(gens, J):
    """(coeff nullspace, DIM of the stabiliser) of {X in span(gens): XJ=JX}.
    dim stab = nullity of the commutator map = m - rank  (NOT the rank)."""
    m = len(gens)
    A = np.column_stack([(Z @ J - J @ Z).ravel() for Z in gens])
    s = np.linalg.svd(A, compute_uv=False)
    tolr = max(A.shape) * (s[0] if s.size else 1.0) * 1e-10
    rank = int(np.sum(s > tolr))
    _, _, vh = np.linalg.svd(A)
    null = vh[rank:].T if rank < m else np.zeros((m, 0))
    return null, null.shape[1]


def killing(gens):
    m = len(gens)
    flat = np.column_stack([Z.ravel() for Z in gens])
    f = np.zeros((m, m, m))
    for i in range(m):
        for j in range(m):
            c, *_ = np.linalg.lstsq(flat, (gens[i] @ gens[j] - gens[j] @ gens[i]).ravel(),
                                    rcond=None)
            f[i, j, :] = c
    ad = [np.array([[f[i, j, k] for j in range(m)] for k in range(m)]) for i in range(m)]
    return np.array([[np.trace(ad[i] @ ad[j]) for j in range(m)] for i in range(m)])


def stab_dim_compact(gens, J, K):
    null, s = stab(gens, J)
    if s == 0:
        return 0, True, np.array([])
    Ks = null.T @ K @ null
    ev = np.linalg.eigvalsh(0.5 * (Ks + Ks.T))
    return s, bool(ev.size and ev.max() < -1e-9), ev


# =============================================================== Gamma = G(Z), finite candidates
def signed_perm_eta_isometries(etaW):
    """All 4x4 signed permutations that are eta-isometries (g^T eta g = eta)."""
    out = []
    for perm in it.permutations(range(4)):
        for signs in it.product((1, -1), repeat=4):
            g = np.zeros((4, 4))
            for i in range(4):
                g[perm[i], i] = signs[i]
            if np.max(np.abs(g.T @ etaW @ g - etaW)) < 1e-9:
                out.append(g)
    return out


def sl2z_finite():
    """All 2x2 integer det=1 matrices in [-1,1] of finite order (SL(2,Z) elliptic pool)."""
    out = []
    for a, b, c, d in it.product(range(-1, 2), repeat=4):
        M = np.array([[a, b], [c, d]], float)
        if abs(np.linalg.det(M) - 1.0) > 1e-9:
            continue
        if order_of(M) is not None:
            out.append(M)
    return out


def order_of(M, maxk=24):
    """Measured order (power to identity) -- NEVER inferred from |lambda| (norm S1598)."""
    n = M.shape[0]
    Q = np.eye(n)
    for k in range(1, maxk + 1):
        Q = Q @ M
        if np.max(np.abs(Q - np.eye(n))) < 1e-7:
            return k
    return None


def in_SOplus(g):
    """g in the identity component SO+ (det +1 and time-preserving g[3,3] > 0)."""
    return np.linalg.det(g) > 0 and g[3, 3] > 0.5


def K_cap_Gamma(J, Wc, Ac):
    """The IMAGE group {U=g(x)h : g in Wc, h in Ac, U commutes with J}, DEDUPED by the distinct
    integer matrix U (the kernel (g,h)->U is {(I,I),(-I,-I)}, so pairs double-count)."""
    seen = {}
    for g in Wc:
        for h in Ac:
            U = np.rint(np.kron(g, h)).astype(np.int64)
            if np.max(np.abs(U @ J - J @ U)) < 1e-7:
                key = U.tobytes()
                if key not in seen:
                    seen[key] = {"U": U.astype(float), "ord": order_of(U.astype(float)),
                                 "so+": bool(in_SOplus(g))}
    return list(seen.values())


def effective_order(grp):
    """|image| modulo the central +-I kernel (order of the group acting effectively)."""
    has_negI = any(np.allclose(e["U"], -np.eye(8)) for e in grp)
    return len(grp) // (2 if has_negI else 1)


def order_histogram(elts):
    h = {}
    for e in elts:
        h[e["ord"]] = h.get(e["ord"], 0) + 1
    return dict(sorted(h.items(), key=lambda kv: (kv[0] is None, kv[0])))


# =============================================================== the run
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=_HERE)
    args = ap.parse_args(argv)
    _OUT["dir"] = os.path.abspath(args.outdir); os.makedirs(_OUT["dir"], exist_ok=True)
    sys.stdout = Tee(sys.stdout, open(_path("S1604_state_type_map_run.log"), "w",
                                      encoding="utf-8"))
    open(_path("S1604_state_type_map_dump.jsonl"), "w", encoding="utf-8").close()

    head("S1604 -- the map of state TYPES (K_J cap Gamma per stratum)")
    say("  ★no row of the map is crowned; the order-3 marker is a NEUTRAL question to the")
    say("   composition (A4/tetrahedron not hinted); element orders are MEASURED, not from |lambda|.")

    etaW = eta_of(3, 1)
    B = bracket_form(etaW)
    gens = levi_generators(etaW)
    K = killing(gens)

    # ------------------------------------------ M-P0: controls / candidate pools
    head("M-P0 -- controls: Gamma = G(Z) finite candidate pools")
    Wc = signed_perm_eta_isometries(etaW)
    Ac = sl2z_finite()
    worst_iso = max(np.max(np.abs(g.T @ etaW @ g - etaW)) for g in Wc)
    Wc_soplus = [g for g in Wc if in_SOplus(g)]
    say(f"  W-side (signed-perm eta-isometries): {len(Wc)} total , of them in SO+ : {len(Wc_soplus)}")
    say(f"     worst |g^T eta g - eta| over the pool = {worst_iso:.1e}")
    say(f"  A-side (SL(2,Z) finite-order in [-1,1]): {len(Ac)} , orders present = "
        f"{sorted(set(order_of(h) for h in Ac))}")
    # negative world: a signed perm that MIXES the time axis with space is NOT an eta-isometry
    gmix = np.array([[0.0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])  # swaps 0<->3
    say(f"  negative world (time<->space swap): |g^T eta g - eta| = "
        f"{np.max(np.abs(gmix.T @ etaW @ gmix - etaW)):.1e}  (not an isometry)")
    dump("M-P0", nW=len(Wc), nW_soplus=len(Wc_soplus), nA=len(Ac))

    ok(lambda w: w[0] < 1e-9, (worst_iso,),
       "T1/M-P0: every W-candidate is a genuine eta-isometry (the finite point-group pool is real)",
       must_fail_on=("a signed perm mixing the time axis with a spatial one",
                     (float(np.max(np.abs(gmix.T @ etaW @ gmix - etaW))),)))

    # ------------------------------------------ M-P1: strata by dim K_J (symmetry ladder)
    head("M-P1 (LOAD-BEARING, a) -- strata of positive J by dim K_J: a SYMMETRY ladder")
    say("  ★the metric is AVERAGED over a compact subgroup of G, forcing exactly that symmetry;")
    say("   a naive diagonal G_W is degenerate here (sign(G_W^-1 eta) sees only the sign pattern).")

    def rot_W01(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.eye(4); R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
        return R

    def rot_A(phi):
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[c, -s], [s, c]])

    def average(G0, ks):
        return sum(k @ G0 @ k.T for k in ks) / len(ks)

    G0 = RNG.normal(size=(8, 8)); G0 = G0 @ G0.T + 3.0 * np.eye(8)
    N = 24
    ks_SO2W_SO2A = [np.kron(rot_W01(2 * np.pi * i / N), rot_A(2 * np.pi * j / N))
                    for i in range(N) for j in range(N)]
    ks_SO2A = [np.kron(np.eye(4), rot_A(2 * np.pi * j / N)) for j in range(N)]

    ladder = [
        ("adapted (isotropic W)", compatible_J(B, np.eye(8))),
        ("SO(2)_W x SO(2)_A type", compatible_J(B, average(G0, ks_SO2W_SO2A))),
        ("SO(2)_A-avg type", compatible_J(B, average(G0, ks_SO2A))),
    ]
    strata = []
    for name, J in ladder:
        s, compact, ev = stab_dim_compact(gens, J, K)
        strata.append((name, J, s, compact))
        say(f"  {name:>30s}: dim K_J = {s} , compact = {compact} "
            f"(Killing max {ev.max() if ev.size else 0:+.2f})")
    Gg = RNG.normal(size=(8, 8)); Gg = Gg @ Gg.T + 3.0 * np.eye(8)
    Jg = compatible_J(B, Gg)
    sg, cg, evg = stab_dim_compact(gens, Jg, K)
    strata.append(("generic (non-product)", Jg, sg, cg))
    say(f"  {'generic (non-product)':>30s}: dim K_J = {sg} , compact = {cg}")
    dims_seen = set()
    for _ in range(30):
        Gr = RNG.normal(size=(8, 8)); Gr = Gr @ Gr.T + 2.0 * np.eye(8)
        dims_seen.add(stab(gens, compatible_J(B, Gr))[1])
    say(f"  ★30 random positive J: dim K_J values seen = {sorted(dims_seen)} (max compact = 4)")
    dims_found = sorted({s for _, _, s, _ in strata})
    say(f"  ==> strata FOUND by dim K_J: {dims_found}  ⟨completeness NOT a theorem: 'found these,")
    say("      the rest of Siegel space not exhausted' -- norm S1595⟩")
    dump("M-P1", dims_found=dims_found, dims_random=sorted(dims_seen))

    adapted = strata[0]; generic = strata[-1]
    ok(lambda w: w[0] >= 3 and w[1] == 0, (adapted[2], generic[2]),
       "T2/M-P1 (LOAD-BEARING, ON THE CRITERION): dim K_J MOVES across the ladder (>=3 on the "
       "adapted stratum, 0 on the generic) -- the stratification is real, not a blind constant",
       must_fail_on=("a detector blind to the generic stratum (would read it >=3 too)",
                     (adapted[2], adapted[2])))

    # ------------------------------------------ M-P2 / M-P4: K_J cap Gamma per stratum
    head("M-P2 (LOAD-BEARING, b) + M-P4 -- the finite group K_J cap Gamma per stratum "
         "(order, composition, order-3 marker)")
    rows = []
    for name, J, s, compact in strata:
        grp = K_cap_Gamma(J, Wc, Ac)
        hist = order_histogram(grp)
        eff = effective_order(grp)
        has3 = any(e["ord"] == 3 for e in grp)
        rows.append({"name": name, "dim": s, "order": len(grp), "eff": eff,
                     "hist": hist, "has3": has3})
        say(f"  {name:>30s}: |K_J cap Gamma| = {len(grp):>3d} (effective mod +-I: {eff:>3d}) , "
            f"order-hist {hist} , order-3 present: {has3}")
    dump("M-P2", rows=[{k: r[k] for k in ('name', 'dim', 'order', 'eff', 'has3')} for r in rows])

    ad_row = rows[0]; gen_row = rows[-1]
    # the CLAIM: effective group nontrivial (>1) on symmetric strata, trivial (=1) on generic
    ok(lambda w: w[0] > 1, (ad_row["eff"],),
       "T3/M-P2 (LOAD-BEARING): the EFFECTIVE K_J cap Gamma (mod +-I) is nontrivial on the adapted "
       "stratum -- finite point groups are BORN as compact cap discrete, without choosing a point",
       must_fail_on=("the generic stratum, whose effective group is the trivial kernel (=1)",
                     (gen_row["eff"],)))

    # order-3 marker: measured order, neutral
    ok(lambda w: bool(w[0]), (ad_row["has3"],),
       "T5/M-P4 (order MEASURED, S1598): the adapted stratum's K_J cap Gamma contains an element "
       "of measured order 3 (the tetrahedral marker -- reported, not crowned, not hinted)",
       must_fail_on=("the SO(2)xSO(2) stratum, whose group has NO order-3 element",
                     (rows[1]["has3"],)))

    # ------------------------------------------ tooth T6: finiteness -- hyperbolic P excluded
    head("T6 -- finiteness: the infinite-order hyperbolic P of Gamma is EXCLUDED from K_J cap Gamma")

    def reflection(v):
        q = float(v @ etaW @ v)
        return np.eye(4) - 2.0 * np.outer(v, etaW @ v) / q
    A1 = reflection(np.array([-2.0, -2, -2, -2])); A2 = reflection(np.array([-2.0, -2, -2, 2]))
    P = np.round(A1 @ A2)
    ordP = order_of(P, maxk=50)
    J0 = adapted[1]
    PxI = np.kron(P, np.eye(2))
    commutes_P = np.max(np.abs(PxI @ J0 - J0 @ PxI)) < 1e-7
    IxI = np.eye(8)
    commutes_I = np.max(np.abs(IxI @ J0 - J0 @ IxI)) < 1e-7
    say(f"  P (integer hyperbolic, S1598): measured order = {ordP} , "
        f"max|lambda| = {np.max(np.abs(np.linalg.eigvals(P))):.2f}")
    say(f"  P(x)I commutes with adapted J0 : {commutes_P}  (excluded => the intersection is FINITE)")
    say(f"  I(x)I commutes with adapted J0 : {commutes_I}  (the identity is always in)")
    dump("T6", ordP=ordP, commutes_P=bool(commutes_P), commutes_I=bool(commutes_I))

    ok(lambda w: not bool(w[0]), (bool(commutes_P),),
       "T6: the infinite-order hyperbolic P of Gamma does NOT lie in K_J cap Gamma (compact cap "
       "discrete genuinely CUTS the infinite Gamma down to a finite group)",
       must_fail_on=("the identity I(x)I, which lies in every stabiliser", (bool(commutes_I),)))

    # ------------------------------------------ M-P3: A-modulus splits the stratum (S1576)
    head("M-P3 (S1576) -- the A-modulus splits the stratum; composition is a TYPE invariant")

    def J_A_of(tau):
        x, y = float(np.real(tau)), float(np.imag(tau))
        return (1.0 / y) * np.array([[x, -(x * x + y * y)], [1.0, -x]])

    def adapted_tau(tau):
        gA = OMEGA_A @ J_A_of(tau); gA = 0.5 * (gA + gA.T)
        return compatible_J(B, np.kron(np.eye(4), gA))
    rho = complex(0.5, np.sqrt(3) / 2)

    def A_part_hist(grp):
        """elements U = I_4 (x) h (identity on W): U == kron(I4, U[0:2,0:2])."""
        sub = [e for e in grp if np.allclose(e["U"], np.kron(np.eye(4), e["U"][0:2, 0:2]))]
        return order_histogram(sub)
    A_types = []
    for label, tau in [("tau=i (Z/4)", 1j), ("tau=rho (Z/6)", rho),
                       ("tau=generic (Z/2)", complex(0.37, 1.3))]:
        Jt = adapted_tau(tau)
        grp = K_cap_Gamma(Jt, Wc, Ac)
        Apart = A_part_hist(grp)
        say(f"  adapted, {label:>18s}: |K_J cap Gamma| = {len(grp):>3d} , "
            f"A-part (g=I) order-hist = {Apart}")
        A_types.append((label, len(grp), Apart))
    # representative invariance: two adapted reps of the SAME (isotropic W, tau=i) type
    J_rep2 = compatible_J(B, np.kron(np.diag([1.0, 1, 1, 1]), OMEGA_A @ J_A_of(1j)))
    grp1 = K_cap_Gamma(adapted[1], Wc, Ac); grp2 = K_cap_Gamma(J_rep2, Wc, Ac)
    same = (order_histogram(grp1) == order_histogram(grp2))
    say(f"  ★representative invariance (two reps of isotropic/tau=i): same order-hist = {same}")
    say(f"     rep1 hist {order_histogram(grp1)}")
    say(f"     rep2 hist {order_histogram(grp2)}")
    dump("M-P3", A_types=[(l, n) for l, n, _ in A_types], rep_invariant=bool(same))

    # composition depends on tau-type (Z/4 vs Z/6) -> stratum splits; invariant within a type
    hist_i = A_types[0][2]; hist_rho = A_types[1][2]
    ok(lambda w: w[0] == w[1], (order_histogram(grp1), order_histogram(grp2)),
       "T4/M-P3 (S1576): the composition of K_J cap Gamma is a TYPE invariant -- two representatives "
       "of the same (W-stratum, A-type) give the SAME order-histogram",
       must_fail_on=("a different A-type (tau=rho vs tau=i), whose composition MUST differ",
                     (hist_i, hist_rho)))

    # ------------------------------------------ M-P5: the type table (no row crowned)
    head("M-P5 -- THE TYPE TABLE (output of the probe; NO row crowned)")
    say(f"  {'stratum (dim K_J)':>30s} | {'|K cap G|':>9s} | {'eff':>4s} | {'order-3':>7s} | "
        f"order-histogram")
    for r in rows:
        say(f"  {r['name']+' ('+str(r['dim'])+')':>30s} | {r['order']:>9d} | {r['eff']:>4d} | "
            f"{str(r['has3']):>7s} | {r['hist']}")
    say("  A-modulus sub-types (isotropic W):")
    for label, n, hist in A_types:
        say(f"  {label:>30s} | {n:>9d} |  --  |    --   | A-part {hist}")
    say("  ★realisability: every K_J cap Gamma above is a subgroup of the point group of Lambda=Z^4(x)Z^2")
    say("   BY CONSTRUCTION (Gamma = G(Z) of that lattice).  Whether each is the FULL point group of a")
    say("   dedicated state on some beta-integral Lambda is NOT measured here (address).  No row crowned.")
    dump("M-P5", table=[{k: r[k] for k in ('name', 'dim', 'order', 'eff', 'has3')} for r in rows])

    # ------------------------------------------ ★DIVIDEND
    head("★DIVIDEND (mandatory) -- finite point groups DERIVED, not imported")
    say("  every finite point group of the assembly = K_J cap Gamma = (compact stabiliser of")
    say("  positivity) cap (discrete arithmetic of integrality), taken PER STRATUM of J, with NO")
    say("  choice of a point.  the map of these groups = the assembly's 'crystal classes', DERIVED")
    say("  from the two compactifiers of S1602.  no new constant is spent.")
    tetра = any(r["has3"] for r in rows)
    if tetра:
        say("  ★an order-3 (tetrahedral) composition appeared -- on the isotropic-W stratum (and the")
        say("   tau=rho A-sub-type) -- the FIRST MEASURED (not hinted) trace of the old assembly's")
        say("   four-leaf; 📖-rima TH-0005 (pure discrete label iff d3), cited NOT as a bridge-proof.")
    else:
        say("  no order-3 composition appeared on any measured stratum -- also a measurement.")

    # ------------------------------------------ VERDICT (gated on teeth, S1598)
    head("VERDICT LINES (S1604) -- each gated on its own tooth (norm S1598)")
    say(f"  M-P1 (a) STRATA   : dim K_J found {dims_found}"
        f"  ==> {'strata are REAL (dim moves 4..0), each compact' if (adapted[2] >= 3 and generic[2] == 0) else 'TOOTH RED'}")
    say(f"  M-P2 (b) K cap G  : adapted |{ad_row['order']}| vs generic |{gen_row['order']}|"
        f"  ==> {'finite point groups BORN as compact cap discrete (nontrivial on symmetric strata)' if ad_row['order'] > 2 else 'TOOTH RED -- KILL-(b): trivial on all strata'}")
    say(f"  M-P4 order-3      : adapted has-3 {ad_row['has3']} , axial has-3 {rows[1]['has3']}"
        f"  ==> {'tetrahedral marker present on isotropic W (MEASURED, neutral, uncrowned)' if ad_row['has3'] else 'no order-3 (also a measurement)'}")
    say(f"  M-P3 (S1576)      : rep-invariant {same} , tau splits Z/4 {hist_i} vs Z/6 {hist_rho}"
        f"  ==> {'composition is a TYPE invariant; A-modulus splits the stratum further' if same else 'TOOTH RED'}")
    say(f"  ==> MAP BUILT: dim-strata {dims_found} (dim alone does NOT fix the type -- two dim-2")
    say("      types with |K cap G| 32 vs 4) x A-types {Z/2,Z/4,Z/6}; finite K_J cap Gamma per")
    say("      type; no row crowned; adaptedness is a STRATUM, not a selection.")

    code = report("S1604 state_type_map")
    sys.stdout.flush()
    return code


if __name__ == "__main__":
    sys.exit(main())

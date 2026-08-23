# -*- coding: utf-8 -*-
"""S1598 -- child-3.1, TACT 7: THE PRICE OF LIFTING A LATTICE INTO THE GROUP OF FLOOR 0.

S1595 closed the finite line by construction: a lattice is an infinite object.  But an infinite
state on THIS floor is not a bare set of points in a vector space -- the parent handed up an
ALGEBRA n0 = (W (x) A) + Z with the bracket beta = -eta|W.  A discrete subgroup of the vector
space forgets the bracket; a discrete subgroup of the GROUP has to live with it.  That
difference is the house-specific content this tact measures.

  (a) the price: the commutator of two translations is a central element carrying the value of
      the bracket, so discreteness forces those values into a discrete subgroup of the centre.
      Is that a condition on Lambda, and which one?
  (b) point groups of the named families, with a tooth ON THE CRITERION itself.
  (c) does (a) + maximality of the point group single out a class canonically?

★The criterion of commensurability carries its own negative world.  A naive real-gcd on
incommensurable numbers runs down to the tolerance and FAKES success -- such a detector would
have printed "the condition is empty" and killed the tact by a false negative.  That norm was
lifted by the court of S1595 from my own address.

Ex-ante: child-3.1/S1598_INFINITE_STATES_EXANTE.md, committed BEFORE this file existed.

Run line (unconditionally into the seal):
    python child-3.1/S1598_infinite_states.py --outdir child-3.1
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
    with open(_path("S1598_infinite_states_dump.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"rec": rec, **payload}, sort_keys=True, default=str) + "\n")


def mat(name, M, fmt="%6.2f"):
    say(f"    {name} =")
    for row in np.atleast_2d(M):
        say("      [" + " ".join(fmt % x for x in row) + "]")


# ================================================= 1. THE FLOOR, REBUILT
def eta_of(p, q):
    return np.diag([1.0] * p + [-1.0] * q)


def bracket_form(etaW):
    """B on W (x) A: B(w(x)a, w'(x)b) = -eta(w,w') * omega(a,b).  Index k = 4*? -> (w, a)."""
    om = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return -np.kron(etaW, om)                       # index = w*2 + a


# ================================================= 2. ★THE CRITERION, WITH ITS OWN WORLDS
def commensurable(values, rel=1e-9, sweep=200):
    """Do all values lie in c*Z for one c > 0?  Returns (verdict, step, worst residual).

    ★The naive real-gcd is exactly the trap named in the ex-ante: on incommensurable numbers it
    runs the step down towards the tolerance and FAKES a yes.  So the step is not chased -- it
    is SEARCHED among the values' own magnitudes and then the residual is measured honestly,
    and a step that has become negligible against the data is REFUSED, not accepted.
    """
    v = np.array([x for x in np.abs(np.ravel(values)) if abs(x) > 1e-12])
    if v.size == 0:
        return True, 0.0, 0.0
    scale = float(np.max(v))
    best = (False, 0.0, np.inf)
    for m in range(1, sweep + 1):
        c = float(np.min(v)) / m
        if c < rel * scale:                          # ★refused: the step became meaningless
            break
        r = float(np.max(np.abs(v / c - np.round(v / c)))) * c / scale
        if r < best[2]:
            best = (r < 1e-8, c, r)
    return best


# ================================================= 3. NAMED FAMILIES
def family_G0():
    return RNG.normal(size=(8, 8))


def family_G1(B):
    """B-integral: Darboux basis of B, in which the values of B are exactly {0, +-1}."""
    n = 8
    M = np.zeros((n, n))
    rem = list(range(n))
    basis, vecs = [], np.eye(n)
    pool = [vecs[:, i] for i in range(n)]
    out = []
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
        out += [e, f]
        pool = [g - float(e @ B @ g) * f + float(f @ B @ g) * e for g in pool]
    M = np.column_stack(out)
    del rem, basis
    return M


def family_G2(LW, LA):
    """The product lattice L_W (x) L_A, in the index convention k = w*2 + a."""
    return np.kron(LW, LA)


# ================================================= 4. INTEGRAL ISOMETRIES OF (Z^4, eta)
def reflection(v, etaW):
    q = float(v @ etaW @ v)
    return np.eye(4) - 2.0 * np.outer(v, etaW @ v) / q


def is_integral(M):
    return float(np.max(np.abs(M - np.round(M)))) < 1e-9


# =============================================================================== 5. THE RUN
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=_HERE)
    args = ap.parse_args(argv)
    _OUT["dir"] = os.path.abspath(args.outdir)
    os.makedirs(_OUT["dir"], exist_ok=True)
    sys.stdout = Tee(sys.stdout, open(_path("S1598_infinite_states_run.log"), "w",
                                      encoding="utf-8"))
    open(_path("S1598_infinite_states_dump.jsonl"), "w", encoding="utf-8").close()

    head("S1598 -- the price of lifting a lattice into the GROUP of floor 0")
    say("  ★an infinite state is represented FINITELY: Lambda = M Z^8, and a point-group")
    say("   element as (g,h) together with U = M^-1 (g (x) h) M in GL(8,Z).")
    say("  ★the word COMPLETE is used nowhere in this probe except where a theorem stands.")

    etaW = eta_of(3, 1)
    B = bracket_form(etaW)
    rk = int(np.linalg.matrix_rank(B, tol=1e-8))
    say(f"\n  B = -eta|W (x) omega_A on W(x)A : antisymmetry "
        f"{float(np.max(np.abs(B + B.T))):.1e} , rank {rk} of 8")
    dump("B", rank=rk)

    ok(lambda w: w[0] == 8, (rk,),
       "T1: the bracket form B on the translations is NON-degenerate (rank 8)",
       must_fail_on=("a deliberately degenerate form",
                     (int(np.linalg.matrix_rank(np.kron(np.diag([1.0, 1, 1, 0]),
                                                        np.array([[0.0, 1], [-1, 0]])),
                                                tol=1e-8)),)))

    # ------------------------------------------ ★T2: the criterion, with BOTH worlds
    head("★THE CRITERION OF COMMENSURABILITY -- published with its own two worlds")
    good = np.array([2.0, 3.0, 5.0]) * 0.37
    bad = np.array([1.0, np.sqrt(2.0)])
    vg, cg, rg = commensurable(good)
    vb, cb, rb = commensurable(bad)
    say(f"  {{2,3,5}}*0.37 : verdict {vg} , step {cg:.6f} , residual {rg:.3e}")
    say(f"  {{1, sqrt 2}}  : verdict {vb} , step {cb:.6f} , residual {rb:.3e}")
    say("  ==> a naive real-gcd would have said YES to the second by driving the step to the")
    say("      tolerance; the step is refused once it becomes negligible against the data.")
    dump("criterion", good=vg, bad=vb, rg=rg, rb=rb)

    ok(lambda w: w[0] is True and w[1] is False, (vg, vb),
       "T2 ★ON THE CRITERION ITSELF (norm S1595, lifted from my own address): the "
       "commensurability test says YES on {2,3,5}c and NO on {1, sqrt 2} -- it can fail",
       must_fail_on=("a world where the test waved the incommensurable pair through",
                     (vg, True)))

    # ------------------------------------------ I-P1: the price of the lift
    head("I-P1 (LOAD-BEARING) -- the price: does a generic Lambda lift to a DISCRETE subgroup?")
    res_G0, res_G1 = [], []
    for _ in range(50):
        M0 = family_G0()
        V0 = M0.T @ B @ M0
        res_G0.append(commensurable(V0)[0])
    M1 = family_G1(B)
    V1 = M1.T @ B @ M1
    v1, c1, r1 = commensurable(V1)
    say(f"  G0 generic  : the values of B on Lambda are commensurable in "
        f"{sum(res_G0)} of {len(res_G0)} draws")
    say(f"  G1 B-integral: verdict {v1} , step {c1:.6f} , residual {r1:.3e}")
    say(f"     (its value matrix has entries {sorted(set(np.round(V1 / c1).astype(int).ravel().tolist()))})")
    say("  ==> the condition is NOT empty: it is exactly  M^T B M  integral up to one factor.")
    dump("price", g0_pass=int(sum(res_G0)), g0_n=len(res_G0), g1=v1, step=float(c1))

    ok(lambda w: w[0] == 0 and w[1] is True, (sum(res_G0), v1),
       "T3/I-P1 (LOAD-BEARING): a generic lattice does NOT lift to a discrete subgroup, while "
       "the B-integral family does -- so the price of the lift is a real condition, and the "
       "house-specific content of the map is non-empty",
       must_fail_on=("a world where the generic lattices passed too", (len(res_G0), v1)))

    # ------------------------------------------ I-P3: the step is input-1, not a new bit
    head("I-P3 -- the step of the centre: a NEW input, or the SAME bit as input-1 of S1590?")
    say(f"  {'t':>6s} {'step c(t Lambda)':>18s} {'ratio to c(Lambda)':>20s} {'t^2':>10s}")
    ratios = []
    for t in (0.5, 2.0, 3.0):
        vt, ct, _ = commensurable((t * M1).T @ B @ (t * M1))
        ratios.append(ct / c1)
        say(f"  {t:>6.2f} {ct:>18.6f} {ct / c1:>20.6f} {t * t:>10.4f}")
    dev = max(abs(r - t * t) for r, t in zip(ratios, (0.5, 2.0, 3.0)))
    say(f"  worst |ratio - t^2| = {dev:.3e}")
    say("  ==> the grading delta of S1590 acts as (Lambda, c) -> (e^t Lambda, e^{2t} c):")
    say("      the step is NOT a new input, it is the SAME bit as input-1 (the scale).")
    dump("step", ratios=ratios, dev=dev)

    ok(lambda w: w[0] < 1e-6, (dev,),
       "T4/I-P3: the step of the centre scales as t^2 while the lattice scales as t -- so it "
       "rides on the scale bit already counted as input-1 in S1590, and is not a new one",
       must_fail_on=("the exponent 1, which must disagree",
                     (max(abs(r - t) for r, t in zip(ratios, (0.5, 2.0, 3.0))),)))

    # ------------------------------------------ I-P4: the product family
    head("I-P4 -- the product family L_W (x) L_A: what the condition reduces to")
    om = np.array([[0.0, 1.0], [-1.0, 0.0]])
    LA = np.array([[1.0, 0.0], [0.0, 1.0]])
    LA2 = RNG.normal(size=(2, 2))
    om_int = [commensurable(L.T @ om @ L)[0] for L in (LA, LA2)]
    say(f"  omega on a RANK-2 lattice lands in det*Z : identity {om_int[0]} , random "
        f"{om_int[1]}  (always, since only one independent value exists)")
    LW_gen = RNG.normal(size=(4, 4))
    LW_int = np.eye(4)
    p_gen = commensurable(family_G2(LW_gen, LA).T @ B @ family_G2(LW_gen, LA))[0]
    p_int = commensurable(family_G2(LW_int, LA).T @ B @ family_G2(LW_int, LA))[0]
    e_gen = commensurable(LW_gen.T @ etaW @ LW_gen)[0]
    e_int = commensurable(LW_int.T @ etaW @ LW_int)[0]
    say(f"  L_W generic : eta|L_W commensurable {e_gen}  ->  the product lifts {p_gen}")
    say(f"  L_W = Z^4   : eta|L_W commensurable {e_int}  ->  the product lifts {p_int}")
    say("  ==> on the product family the price reduces EXACTLY to: eta|L_W integral up to a "
        "factor.")
    dump("product", om_int=om_int, p_gen=p_gen, p_int=p_int, e_gen=e_gen, e_int=e_int)

    ok(lambda w: w[0] is True and w[1] is True and w[2] is False and w[3] is False,
       (om_int[0], om_int[1], e_gen, p_gen),
       "T5/I-P4: omega on a rank-two lattice is always commensurable, while a generic eta|L_W "
       "is not -- so the whole price sits on the W factor",
       must_fail_on=("Z^4, where the W factor passes and so does the product",
                     (om_int[0], om_int[1], e_int, p_int)))

    # ------------------------------------------ I-P5: the continuous part
    head("I-P5 -- the CONTINUOUS part of the stabiliser of a full-rank Lambda: verified, not "
         "postulated")
    soW = []
    for i in range(4):
        for j in range(i + 1, 4):
            E = np.zeros((4, 4))
            E[i, j], E[j, i] = 1.0, 1.0
            A_ = np.outer(np.eye(4)[i], etaW @ np.eye(4)[j]) - np.outer(np.eye(4)[j],
                                                                       etaW @ np.eye(4)[i])
            soW.append(A_)
            del E
    sl2 = [np.array([[1.0, 0.0], [0.0, -1.0]]), np.array([[0.0, 1.0], [0.0, 0.0]]),
           np.array([[0.0, 0.0], [1.0, 0.0]])]
    gens = [np.kron(S, np.eye(2)) for S in soW] + [np.kron(np.eye(4), T) for T in sl2]
    Minv = np.linalg.inv(M1)
    worst_far, zero_far = 0.0, 0.0
    for G in gens:
        for t in (1e-3, 1e-2, 1e-1):
            U = Minv @ (np.eye(8) + t * G + 0.5 * t * t * G @ G) @ M1
            worst_far = max(worst_far, float(np.max(np.abs(U - np.round(U)))))
    U0 = Minv @ np.eye(8) @ M1
    zero_far = float(np.max(np.abs(U0 - np.round(U0))))
    say(f"  for every non-zero generator X of the Levi and small t: distance of "
        f"M^-1 exp(tX) M to the nearest INTEGER matrix = {worst_far:.3e}")
    say(f"  ★moving contrast: at X = 0 the same distance is {zero_far:.3e}")
    say("  ==> no one-parameter subgroup preserves Lambda: the continuous part is 0.")
    dump("continuous", worst=worst_far, zero=zero_far)

    ok(lambda w: w[0] > 1e-6 and w[1] < 1e-9, (worst_far, zero_far),
       "T6/I-P5: the continuous part of the stabiliser of a full-rank lattice is zero -- "
       "MEASURED against a moving contrast, not postulated",
       must_fail_on=("a world where even X = 0 was counted as moving",
                     (worst_far, worst_far)))

    # ------------------------------------------ I-P6: the vertex question
    head("I-P6 (THE VERTEX) -- is the point group finite?  If it is not, maximality selects "
         "nothing")
    say("  ★the witness is SEARCHED, not guessed: my first guess (reflections in e1 and in")
    say("   (1,1,1,1)) turned out to have order 4, and the order detector caught it -- that")
    say("   failed guess is kept below as the negative world.  See the seal §1.")
    say("  named convention of the search (introduced at probe time, said out loud): integral")
    say("  reflections in vectors with entries in -2..2; a POSITIVE find needs no completeness.")
    import itertools as _it

    def order_of(P, maxk=400):
        """The ORDER, measured -- never inferred from the modulus of an eigenvalue."""
        Q = np.eye(4)
        for k in range(1, maxk + 1):
            Q = Q @ P
            if float(np.max(np.abs(Q - np.eye(4)))) < 1e-8:
                return k
        return None

    refls = []
    for v in _it.product(range(-2, 3), repeat=4):
        v = np.array(v, float)
        if not v.any():
            continue
        if abs(float(v @ etaW @ v)) < 1e-9:
            continue
        R = reflection(v, etaW)
        if is_integral(R) and float(np.max(np.abs(R.T @ etaW @ R - etaW))) < 1e-9:
            refls.append((tuple(int(x) for x in v), R))
    say(f"  integral reflections in the named box : {len(refls)}")
    found = None
    for (va, A) in refls:
        for (vb, Bm) in refls:
            Pp = A @ Bm
            m = float(np.max(np.abs(np.linalg.eigvals(Pp))))
            if m > 1.0 + 1e-6:
                found = (va, vb, Pp, m)
                break
        if found:
            break
    va, vb, P, maxev = found
    ordP = order_of(P)
    say(f"  witness : reflection in {va} composed with reflection in {vb}")
    mat("P", P, fmt="%5.0f")
    say(f"  P integral {is_integral(P)} , isometry "
        f"{float(np.max(np.abs(P.T @ etaW @ P - etaW))):.1e} , max |lambda| = {maxev:.6f} , "
        f"measured order = {ordP}")
    # the failed guess, kept as the negative world
    Q1 = reflection(np.array([1.0, 0.0, 0.0, 0.0]), etaW)
    Q2 = reflection(np.array([1.0, 1.0, 1.0, 1.0]), etaW)
    Pbad = Q2 @ Q1
    ord_bad = order_of(Pbad)
    say(f"  ★negative world (my first guess): reflections in e1 and (1,1,1,1) give max |lambda| "
        f"= {float(np.max(np.abs(np.linalg.eigvals(Pbad)))):.6f} and a FINITE order {ord_bad}")
    order_inf = (ordP is None) and (maxev > 1.0 + 1e-6)
    say(f"  ==> the point group of Z^4 (x) Z^2 contains an element of infinite order : "
        f"{order_inf}   ⟹ the group is INFINITE")
    say("  ==> 'maximality of the point group' selects NOTHING here: maximal is not finite, and")
    say("      the infinite part is exactly what a CHOSEN DIRECTION would have to kill --")
    say("      and that choice is input-2 of S1590 (the frame).")
    dump("vertex", va=va, vb=vb, maxev=maxev, order=ordP, order_bad=ord_bad,
         infinite=bool(order_inf))

    ok(lambda w: w[0] is None and w[1] > 1.0 + 1e-6 and w[2] is True and w[3] < 1e-9,
       (ordP, maxev, is_integral(P), float(np.max(np.abs(P.T @ etaW @ P - etaW)))),
       "T7/I-P6: the witness is a genuine INTEGRAL isometry of (Z^4, eta) whose ORDER is "
       "measured to be infinite (and whose spectral radius exceeds 1) -- the point group is "
       "INFINITE",
       must_fail_on=("my own first guess, whose order is finite",
                     (ord_bad, float(np.max(np.abs(np.linalg.eigvals(Pbad)))),
                      is_integral(Pbad),
                      float(np.max(np.abs(Pbad.T @ etaW @ Pbad - etaW))))))

    # ---------------------------------------------------------------------------- VERDICT
    head("VERDICT LINES (S1598)")
    say(f"  I-P1 PRICE     : generic Lambda lifts in {sum(res_G0)} of {len(res_G0)} ; the "
        f"B-integral family lifts (step {c1:.4f}) ==> the condition is NOT empty")
    say("                   and it is exactly: M^T B M integral up to one common factor")
    say(f"  I-P2 CRITERION : {{2,3,5}}c -> True , {{1, sqrt2}} -> False  (the test can fail)")
    say(f"  I-P3 STEP      : c(t Lambda)/c(Lambda) = t^2 (worst deviation {dev:.1e}) ==> the "
        "step rides on input-1 of S1590, it is NOT a new bit")
    say(f"  I-P4 PRODUCT   : omega always commensurable ; the price reduces to eta|L_W integral")
    say(f"  I-P5 CONTINUOUS: 0, measured ({worst_far:.1e} against {zero_far:.1e})")
    say(f"  I-P6 VERTEX    : witness of INFINITE order (measured: {ordP}) with |lambda|max = "
        f"{maxev:.4f} ==> the point group is INFINITE ==> maximality selects NOTHING without a "
        "chosen direction  ⟨my first guess had order {ob}⟩".replace("{ob}", str(ord_bad)))
    say("  ==> KILL-(iii) FIRED, as carved before the run: no canonical road without a "
        "functional; the fork returns to the author.")

    code = report("S1598 infinite_states")
    sys.stdout.flush()
    return code


if __name__ == "__main__":
    sys.exit(main())

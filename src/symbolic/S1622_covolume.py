#!/usr/bin/env python3
# author of the probe: A (lane-A), S1622, naryad #13 takt-21 (covolume by number).
# Covolume of O^+(I_{3,1}) as a NUMBER: verify the Vinberg simple roots -> Coxeter diagram [3,4,4],
# compute the orthoscheme volume via the Lobachevsky function (Vinberg/Kellerhals formula),
# predict orbit density 1/(|Stab(e4)|*V), and reproduce S1602's direct count natively.
# Lobachevsky via mpmath.clsin(2,2t)/2.  Reader NJ (reference): V=0.0763304662, rho=0.27294.
# RUN LINE (unconditional):  python child-3.1/S1622_covolume.py --outdir child-3.1
import argparse, os, json
import mpmath as mp
import numpy as np
from math import gcd, isqrt, cosh, sinh, pi, acos, sqrt

mp.mp.dps = 30
J = [1, 1, 1, -1]

def dot(x, y):
    return sum(J[i] * x[i] * y[i] for i in range(4))

def reflection_integral(r):
    """s_r(x)=x - 2(x.r)/(r.r) r integral for all basis e_i?"""
    nr = dot(r, r)
    for k in range(4):
        e = [1 if i == k else 0 for i in range(4)]
        coeff = 2 * dot(e, r)
        if coeff % nr != 0:
            return False
        img = [e[i] - (coeff // nr) * r[i] for i in range(4)]
        if any(not float(c).is_integer() for c in img):
            return False
    return True

def lobach(theta):
    return mp.clsin(2, 2 * theta) / 2

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="."); ap.add_argument("--Rmax", type=int, default=7)
    args = ap.parse_args()
    print("=" * 74)
    print("S1622 — covolume of O^+(I_{3,1}) by number: Vinberg -> [3,4,4] -> Lobachevsky volume")
    print("=" * 74)

    roots = {"r1": (1, -1, 0, 0), "r2": (0, 1, -1, 0), "r3": (0, 0, 1, 0), "r4": (-1, -1, -1, 1)}
    order = ["r1", "r2", "r3", "r4"]

    # ---- K-0a: roots valid ----
    print("\n-- roots (Vinberg simple roots, controlling vector e4) --")
    roots_ok = True
    for k in order:
        r = roots[k]; nr = dot(r, r); g = 0
        for c in r: g = gcd(g, abs(c))
        integ = reflection_integral(r)
        prim = (g == 1)
        ok = (nr in (1, 2)) and integ and prim
        roots_ok = roots_ok and ok
        print("   %s = %-14s norm=%d  reflection integral=%s primitive=%s  %s" %
              (k, str(r), nr, integ, prim, "OK" if ok else "FAIL"))

    # ---- Coxeter diagram from root Gram ----
    print("\n-- Coxeter diagram (m_ij from cos = -(ri.rj)/sqrt(ni nj)) --")
    labels = {}
    for i in range(4):
        for j in range(i + 1, 4):
            ri, rj = roots[order[i]], roots[order[j]]
            c = -dot(ri, rj) / sqrt(dot(ri, ri) * dot(rj, rj))
            if abs(c) < 1e-12:
                m = 2
            elif abs(c - 1) < 1e-12:
                m = "inf"
            else:
                m = int(round(pi / acos(c)))
            labels[(i, j)] = m
    print("   r1-r2: m=%s  r2-r3: m=%s  r3-r4: m=%s   (chain labels)" %
          (labels[(0, 1)], labels[(1, 2)], labels[(2, 3)]))
    print("   r1-r3: m=%s  r1-r4: m=%s  r2-r4: m=%s   (should be 2 = orthogonal)" %
          (labels[(0, 2)], labels[(0, 3)], labels[(1, 3)]))
    chain = [labels[(0, 1)], labels[(1, 2)], labels[(2, 3)]]
    offchain = [labels[(0, 2)], labels[(0, 3)], labels[(1, 3)]]
    diagram_ok = (chain == [3, 4, 4] and offchain == [2, 2, 2])
    print("   diagram = [%s] linear : %s  -> [3,4,4]: %s" %
          (",".join(map(str, chain)), all(x == 2 for x in offchain),
           "OK" if diagram_ok else "FAIL"))

    # signature of cosine Gram (should be (3,1) Lorentzian)
    CG = np.array([[1.0 if i == j else -dot(roots[order[i]], roots[order[j]]) /
                    sqrt(dot(roots[order[i]], roots[order[i]]) * dot(roots[order[j]], roots[order[j]]))
                    for j in range(4)] for i in range(4)])
    eig = np.linalg.eigvalsh(CG)
    npos = int((eig > 1e-9).sum()); nneg = int((eig < -1e-9).sum())
    print("   cosine-Gram signature = (%d,%d)  (expect (3,1) hyperbolic): %s" %
          (npos, nneg, "OK" if (npos, nneg) == (3, 1) else "FAIL"))

    # ---- K-0b: Lobachevsky control ----
    L_pi6 = float(lobach(mp.pi / 6)); L_pi4 = float(lobach(mp.pi / 4))
    lob_ok = (abs(L_pi6 - 0.5074708) < 1e-6 and abs(L_pi4 - 0.4579819) < 1e-6)
    print("\n-- Lobachevsky control: L(pi/6)=%.7f (0.5074708) L(pi/4)=%.7f (0.4579819) : %s" %
          (L_pi6, L_pi4, "OK" if lob_ok else "FAIL"))

    # ---- orthoscheme volume (alpha=pi/3, beta=pi/4, gamma=pi/4) ----
    a = mp.pi / 3; b = mp.pi / 4; g = mp.pi / 4
    delta = mp.atan(mp.sqrt(mp.cos(b) ** 2 - mp.sin(a) ** 2 * mp.sin(g) ** 2) / (mp.cos(a) * mp.cos(g)))
    V = mp.mpf(1) / 4 * (lobach(a + delta) - lobach(a - delta)
                         + lobach(g + delta) - lobach(g - delta)
                         - lobach(mp.pi / 2 - b + delta) + lobach(mp.pi / 2 - b - delta)
                         + 2 * lobach(mp.pi / 2 - delta))
    Vf = float(V)
    reader_V = 0.0763304662
    V_ok = abs(Vf - reader_V) < 1e-7
    print("\n-- ORTHOSCHEME VOLUME --")
    print("   delta = %.10f (expect pi/4 = %.10f)" % (float(delta), pi / 4))
    print("   V(covolume) = %.10f   reader NJ 0.0763304662   coincide: %s" % (Vf, V_ok))

    # ---- density prognosis ----
    stab = 48
    rho_pred = 1.0 / (stab * Vf)
    print("\n-- DENSITY PROGNOSIS: 1/(|Stab(e4)|*V) = 1/(48*%.8f) = %.8f  (reader 0.27294)" %
          (Vf, rho_pred))

    # ---- direct count (native reproduction of S1602) ----
    def vol_ball(R):
        return pi * sinh(2 * R) - 2 * pi * R

    def count_N_exact(R):
        vmax = int(cosh(R))
        N = 0
        for v4 in range(1, vmax + 1):
            m = v4 * v4 - 1
            if m == 0:
                N += 1; continue
            A = isqrt(m)
            for av in range(-A, A + 1):
                rem = m - av * av
                if rem < 0:
                    continue
                B = isqrt(rem)
                bs = np.arange(-B, B + 1)
                rem2 = rem - bs * bs
                s = np.floor(np.sqrt(rem2) + 0.5).astype(np.int64)
                exact = (s * s == rem2)
                cnt = np.where(s[exact] > 0, 2, 1)
                N += int(cnt.sum())
        return N

    print("\n-- DIRECT COUNT (native S1602): N(R)=#{v.v=-1, v4<=cosh R}, rho=N/vol(B_R) --")
    print("   R    vol(B_R)     N(R)     rho      (S1602/limit)")
    rows = []
    for R in range(2, args.Rmax + 1):
        vb = vol_ball(R)
        N = count_N_exact(R)
        rho = N / vb
        rows.append((R, vb, N, rho))
        tag = ""
        if R == 4:
            tag = " <- S1602: N=1345 rho=0.2888"
        print("   %d   %10.2f   %6d   %.5f%s" % (R, vb, N, rho, tag))

    # K-1 check: R=4 -> N=1345
    N4 = next(N for (R, vb, N, rho) in rows if R == 4)
    k1 = (N4 == 1345)
    print("   K-1 (native S1602 R=4): N=%d expect 1345 -> %s" % (N4, "PASS" if k1 else "FAIL"))

    rho_last = rows[-1][3]; Rlast = rows[-1][0]
    rel = abs(rho_pred - rho_last) / rho_pred
    kill_B = rel > 0.05
    print("\n-- KILL-B (object match): |rho_pred - rho(R=%d)|/rho_pred = %.4f (>5%%: %s)" %
          (Rlast, rel, "TRIGGERED" if kill_B else "NOT triggered"))
    if not kill_B:
        print("   => O^+(I_{3,1}) = reflection group with [3,4,4] orthoscheme fundamental")
        print("      (no hidden symmetry); density S1602 confirmed by covolume number.")

    verdict = roots_ok and diagram_ok and (npos, nneg) == (3, 1) and lob_ok and V_ok and k1 and not kill_B
    print("\n-- VERDICT --")
    for name, ok in [("roots valid", roots_ok), ("diagram [3,4,4]", diagram_ok),
                     ("signature (3,1)", (npos, nneg) == (3, 1)), ("Lobachevsky ctrl", lob_ok),
                     ("V=0.0763304662", V_ok), ("K-1 S1602 R=4=1345", k1),
                     ("density match (kill-B not fired)", not kill_B)]:
        print("   [%s] %s" % ("OK" if ok else "XX", name))
    print("\n   TEETH: %d/7 -> %s" %
          (sum([roots_ok, diagram_ok, (npos, nneg) == (3, 1), lob_ok, V_ok, k1, not kill_B]),
           "VERDICT OK" if verdict else "VERDICT BROKEN"))

    os.makedirs(args.outdir, exist_ok=True)
    dump = os.path.join(args.outdir, "S1622_covolume_dump.jsonl")
    with open(dump, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "roots", "ok": roots_ok,
                            "chain": chain, "offchain": offchain,
                            "signature": [npos, nneg]}) + "\n")
        f.write(json.dumps({"kind": "volume", "V": Vf, "delta": float(delta),
                            "reader_V": reader_V, "rho_pred": rho_pred}) + "\n")
        for (R, vb, N, rho) in rows:
            f.write(json.dumps({"kind": "density", "R": R, "vol": vb, "N": N, "rho": rho}) + "\n")
    print("\n[dump] %s" % dump)

if __name__ == "__main__":
    main()

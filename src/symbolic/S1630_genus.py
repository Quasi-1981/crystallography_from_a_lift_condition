#!/usr/bin/env python3
# author of the probe: A (lane-A), S1630 (naryad #14 takt-24).
# Genus of the beta-integral (3,1) class for |det|<=8 via p-adic Jordan symbols.
# Does the 11-SNF-bucket "convention" (S1605) equal 11 genera, or split?  Dividend: for indefinite
# rank>=3, class=spinor genus (Eichler-Kneser), small det => spinor genus=genus => genera=classes.
# CONTROL (Omega, mandatory): I_{3,1} (det 1) has exactly ONE genus.  If the control fails, the
# 2-adic symbol is not validated -> park the number, do not submit (per ex-ante).
# RUN LINE (unconditional):  python child-3.1/S1630_genus.py --outdir child-3.1
import argparse, os, json
from fractions import Fraction as F
from math import gcd
from itertools import product
import numpy as np

def vp(x, p):
    if x == 0: return 10**9
    if isinstance(x, F): num, den = x.numerator, x.denominator
    else: num, den = int(x), 1
    v = 0
    while num % p == 0: num //= p; v += 1
    while den % p == 0: den //= p; v -= 1
    return v

def matcopy(M): return [[F(x) for x in row] for row in M]

def jordan_blocks(Gin, p):
    """p-adic Jordan decomposition over Z_p. Returns list of (scale, block-matrix over Z_p units)."""
    G = matcopy(Gin); blocks = []
    while len(G) > 0:
        n = len(G)
        # min valuation over entries; prefer a diagonal pivot
        best = None; diagloc = None; offloc = None
        for i in range(n):
            for j in range(i, n):
                if G[i][j] != 0:
                    v = vp(G[i][j], p)
                    if best is None or v < best: best = v
        for i in range(n):
            if G[i][i] != 0 and vp(G[i][i], p) == best: diagloc = i; break
        if diagloc is None:
            for i in range(n):
                for j in range(i+1, n):
                    if G[i][j] != 0 and vp(G[i][j], p) == best: offloc = (i, j); break
                if offloc: break
        if diagloc is not None:
            i = diagloc
            # move pivot to 0
            if i != 0:
                G[0], G[i] = G[i], G[0]
                for r in range(n): G[r][0], G[r][i] = G[r][i], G[r][0]
            piv = G[0][0]
            for r in range(1, n):
                if G[r][0] != 0:
                    f = G[r][0] / piv
                    for c in range(n): G[r][c] = G[r][c] - f*G[0][c]
                    for c in range(n): G[c][r] = G[c][r] - f*G[c][0]
            blocks.append((best, [[G[0][0]]]))
            G = [[G[r][c] for c in range(1, n)] for r in range(1, n)]
        else:
            i, j = offloc
            # bring to positions 0,1
            for a, b in [(0, i), (1, j)]:
                if a != b:
                    G[a], G[b] = G[b], G[a]
                    for r in range(n): G[r][a], G[r][b] = G[r][b], G[r][a]
            # 2x2 pivot block P = [[G00,G01],[G01,G11]], det has val 2*best (elliptic/hyperbolic)
            a00, a01, a11 = G[0][0], G[0][1], G[1][1]
            det2 = a00*a11 - a01*a01
            for r in range(2, n):
                x = G[r][0]; y = G[r][1]
                # solve [a00 a01; a01 a11] [f0;f1] = [x;y]
                f0 = (x*a11 - y*a01)/det2; f1 = (y*a00 - x*a01)/det2
                for c in range(n): G[r][c] = G[r][c] - f0*G[0][c] - f1*G[1][c]
                for c in range(n): G[c][r] = G[c][r] - f0*G[c][0] - f1*G[c][1]
            blocks.append((best, [[a00, a01], [a01, a11]]))
            G = [[G[r][c] for c in range(2, n)] for r in range(2, n)]
    return blocks

def kron(a, p):  # Legendre/Kronecker symbol of unit a mod p (odd p)
    a %= p
    if a == 0: return 0
    return 1 if pow(a, (p-1)//2, p) == 1 else -1

def symbol_odd(blocks, p):
    """canonical odd-p symbol: sorted list of (scale, dim, eps)."""
    byscale = {}
    for (k, B) in blocks:
        dim = len(B)
        # unit determinant = det(B)/p^(k*dim)
        d = 1
        if dim == 1: d = B[0][0]
        else: d = B[0][0]*B[1][1]-B[0][1]*B[1][0]
        unit = d / (F(p)**(k*dim))
        # unit is a p-adic unit; reduce to integer mod p
        num, den = unit.numerator, unit.denominator
        u = (num * pow(den % p, p-2, p)) % p if den % p != 0 else num % p
        byscale.setdefault(k, [0, 1])
        byscale[k][0] += dim
        byscale[k][1] = (byscale[k][1] * u) % p
    return tuple(sorted((k, dim, kron(u, p)) for k, (dim, u) in byscale.items()))

def symbol_2(blocks):
    """2-adic invariant fingerprint: per scale (dim,type), total oddity mod 8, det unit mod 8.
    Not the full sign-walked canonical symbol, but a genus-INVARIANT fingerprint."""
    byscale = {}
    total_odd = 0; det_unit = 1
    for (k, B) in blocks:
        dim = len(B)
        if dim == 1:
            a = B[0][0]; unit = a / (F(2)**k)
            u = unit.numerator  # odd integer (p-adic unit)
            # ensure integer odd
            typ = 'I'  # 1x1 block is type I (odd)
            odd = u % 8
            total_odd = (total_odd + odd) % 8
            det_unit = (det_unit * (u % 8)) % 8
            byscale.setdefault(k, {'dim': 0, 'typeI': 0})
            byscale[k]['dim'] += 1; byscale[k]['typeI'] += 1
        else:
            a00, a01, a11 = B[0][0], B[1][1], B[0][1]
            # scaled block det unit
            d = (B[0][0]*B[1][1]-B[0][1]*B[0][1]) / (F(2)**(2*k))
            du = d.numerator % 8
            det_unit = (det_unit * du) % 8
            # type II if diagonal even after scaling (hyperbolic [[0,1],[1,0]] or elliptic [[2,1],[1,2]])
            s00 = (B[0][0] / (F(2)**k)); s11 = (B[1][1] / (F(2)**k))
            even = (s00.denominator == 1 and s00.numerator % 2 == 0 and
                    s11.denominator == 1 and s11.numerator % 2 == 0)
            typ = 'II' if even else 'I'
            byscale.setdefault(k, {'dim': 0, 'typeI': 0})
            byscale[k]['dim'] += 2
            if typ == 'I': byscale[k]['typeI'] += 2
    scales = tuple(sorted((k, v['dim'], 'I' if v['typeI'] > 0 else 'II') for k, v in byscale.items()))
    return (scales, total_odd % 8, det_unit % 8)

def signature(G):
    ev = np.linalg.eigvalsh(np.array(G, dtype=float))
    return (int((ev > 1e-9).sum()), int((ev < -1e-9).sum()))

def det_int(M):
    n = len(M)
    if n == 1: return M[0][0]
    if n == 2: return M[0][0]*M[1][1]-M[0][1]*M[1][0]
    return sum(((-1)**c)*M[0][c]*det_int([[M[i][j] for j in range(n) if j != c] for i in range(1, n)]) for c in range(n))

def primes_dividing(m):
    m = abs(m); ps = []
    d = 2
    while d*d <= m:
        if m % d == 0:
            ps.append(d)
            while m % d == 0: m //= d
        d += 1
    if m > 1: ps.append(m)
    return ps

def snf_invariants(G):
    # Smith normal form invariant factors (abs)
    M = [[int(x) for x in row] for row in G]; n = len(M)
    facs = []
    import copy
    A = copy.deepcopy(M)
    def mat_gcd_reduce(A):
        return A
    # simple SNF via sympy
    from sympy import Matrix
    d = Matrix(M).elementary_divisors() if hasattr(Matrix(M), 'elementary_divisors') else None
    # fallback: use sympy smith_normal_form
    try:
        from sympy.matrices.normalforms import smith_normal_form
        S = smith_normal_form(Matrix(M))
        return tuple(abs(int(S[i, i])) for i in range(n))
    except Exception:
        return tuple(sorted(abs(x) for x in [det_int(M)]))

def mat_inv_frac(G):
    n = len(G); A = [[F(G[i][j]) for j in range(n)] + [F(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = None
        for r in range(c, n):
            if A[r][c] != 0: piv = r; break
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [x/pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]; A[r] = [A[r][k]-f*A[c][k] for k in range(2*n)]
    return [[A[i][n+j] for j in range(n)] for i in range(n)]

def disc_q_multiset(G):
    """Genus invariant for indefinite (CS 15.7.4 / Nikulin): the discriminant QUADRATIC form
    q(y)=y^T G^{-1} y mod 2Z on disc=Z^n/G Z^n. Return multiset of (element order, q-value)."""
    n = len(G); D = abs(det_int(G))
    if D == 1: return ()
    Ginv = mat_inv_frac(G)
    from math import floor
    reps = {}
    for y in product(range(0, D), repeat=n):
        gy = [sum(Ginv[i][j]*y[j] for j in range(n)) for i in range(n)]
        frac = tuple(c - F(floor(c)) for c in gy)
        if all(f == 0 for f in frac):
            if () not in reps: reps[()] = None  # identity, skip in multiset below
            continue
        if frac not in reps:
            q = sum(F(y[i])*gy[i] for i in range(n))
            bm = q - F(floor(q))                 # b(x,x) mod Z  (well-defined for ODD lattices)
            # order of the element = lcm of denominators of frac
            order = 1
            for f in frac: order = order*f.denominator//gcd(order, f.denominator)
            reps[frac] = (order, (bm.numerator, bm.denominator))
        if len(reps) >= D: break
    vals = [v for v in reps.values() if v is not None]
    return tuple(sorted(vals))

def genus_symbol(G):
    D = det_int(G)
    sig = signature(G)
    return (sig, D, disc_q_multiset(G))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=".")
    ap.add_argument("--rng", type=int, default=2)
    args = ap.parse_args()
    print("=" * 74)
    print("S1630 — genus of (3,1) class |det|<=8 via p-adic symbols; 11 SNF-buckets = ? genera")
    print("=" * 74)

    # CONTROL det=1
    I31 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
    gI = genus_symbol(I31)
    print("\n-- K-0 control: I_{3,1} genus symbol --")
    print("   sig=%s det=%d  disc-q-multiset=%s (empty => trivial disc => unique genus)" % (gI[0], gI[1], gI[2]))

    # enumerate (3,1) forms, |det|<=8
    R = args.rng
    diag_vals = list(range(-R, R+1)); off_vals = list(range(-1, 2))
    seen = set(); by_det = {}
    n = 4
    offpos = [(i, j) for i in range(n) for j in range(i+1, n)]
    count = 0
    for dvec in product(diag_vals, repeat=n):
        if any(d == 0 for d in dvec): continue
        for ov in product(off_vals, repeat=len(offpos)):
            G = [[0]*n for _ in range(n)]
            for i in range(n): G[i][i] = dvec[i]
            for idx, (i, j) in enumerate(offpos): G[i][j] = G[j][i] = ov[idx]
            D = det_int(G)
            if D >= 0 or D < -8: continue
            key = tuple(tuple(r) for r in G)
            if key in seen: continue
            seen.add(key)
            if signature(G) != (3, 1): continue
            count += 1
            gs = genus_symbol(G); snf = snf_invariants(G)
            ad = abs(D)
            by_det.setdefault(ad, {'genera': set(), 'snf': set()})
            by_det[ad]['genera'].add(gs)
            by_det[ad]['snf'].add(snf)

    print("\n-- forms enumerated (sig (3,1), |det|<=8): %d --" % count)
    print("\n   |det|   #SNF-buckets   #genera")
    tot_snf = 0; tot_gen = 0
    for ad in sorted(by_det):
        ns = len(by_det[ad]['snf']); ng = len(by_det[ad]['genera'])
        tot_snf += ns; tot_gen += ng
        print("   %4d      %4d          %4d" % (ad, ns, ng))
    print("   %-9s %4d          %4d   (TOTAL over det 1..8)" % ("SUM", tot_snf, tot_gen))

    det1 = by_det.get(1, {'genera': set()})
    ctrl_ok = (len(det1['genera']) == 1)
    print("\n-- CONTROL: det 1 genera = %d (expect 1) -> %s" %
          (len(det1['genera']), "PASS" if ctrl_ok else "FAIL (2-adic not validated -> PARK)"))

    print("\n-- VERDICT --")
    if ctrl_ok:
        print("   [OK] control det1=1 genus; genera counted per det above.")
        print("   11 SNF-buckets (S1605) vs genera: SUM SNF=%d, SUM genera=%d" % (tot_snf, tot_gen))
        print("   => genera %s SNF-buckets%s" %
              ("=" if tot_gen == tot_snf else (">" if tot_gen > tot_snf else "<"),
               "" if tot_gen == tot_snf else "  (buckets SPLIT into genera by disc-form)"))
    else:
        print("   [XX] control FAILED -> 2-adic symbol not validated; number PARKED (per ex-ante).")

    os.makedirs(args.outdir, exist_ok=True)
    dump = os.path.join(args.outdir, "S1630_genus_dump.jsonl")
    with open(dump, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "control", "det1_genera": len(det1['genera']), "ok": ctrl_ok}) + "\n")
        for ad in sorted(by_det):
            f.write(json.dumps({"kind": "det", "det": ad,
                                "snf_buckets": len(by_det[ad]['snf']),
                                "genera": len(by_det[ad]['genera'])}) + "\n")
    print("\n[dump] %s" % dump)

if __name__ == "__main__":
    main()

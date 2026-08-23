#!/usr/bin/env python3
# author of the probe: A (lane-A), S1633 (S1629 ver:2, naryad #14 takt-25, reader-3).
# Fix of the split-disc filter: exact Z/n generator (search over cosets, guaranteed complete -- no
# small-box false-negative), require a in (Z/n)^unit, then a QR test.  Re-derives n_min and compares
# to S1629/takt-18, and compares old box-filter vs new exact filter on every Gram (Kill-V).
# RUN LINE (unconditional):  python child-3.1/S1633_nmin_exact_ver2.py --outdir child-3.1
import argparse, os, json
from fractions import Fraction as F
from math import gcd, isqrt, floor
from itertools import product, combinations_with_replacement

def det_int(M):
    n = len(M)
    if n == 1: return M[0][0]
    if n == 2: return M[0][0]*M[1][1]-M[0][1]*M[1][0]
    if n == 3:
        return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
                - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
                + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))
    return sum(((-1)**c)*M[0][c]*det_int([[M[i][j] for j in range(n) if j != c] for i in range(1, n)]) for c in range(n))

def inv_diag(G):
    n = len(G); d = det_int(G)
    return [F(det_int([[G[r][c] for c in range(n) if c != i] for r in range(n) if r != i]), d) for i in range(n)]

def mat_inv_frac(G):
    n = len(G)
    A = [[F(G[i][j]) for j in range(n)] + [F(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]; A[c] = [x/pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]; A[r] = [A[r][k]-f*A[c][k] for k in range(2*n)]
    return [[A[i][n+j] for j in range(n)] for i in range(n)]

def qform(G, x): return sum(G[i][j]*x[i]*x[j] for i in range(len(G)) for j in range(len(G)))
def bil(G, x, y): return sum(G[i][j]*x[i]*y[j] for i in range(len(G)) for j in range(len(G)))

def short_vectors(G, N):
    n = len(G); gd = inv_diag(G); rr = []
    for j in range(n):
        val = F(N)*gd[j]; v = val.numerator//val.denominator if val > 0 else 0
        rr.append(range(-(isqrt(int(v))+1), isqrt(int(v))+2))
    return [x for x in product(*rr) if qform(G, x) == N]

def aut_matrices(G):
    p = len(G); Nc = [G[i][i] for i in range(p)]; cand = [short_vectors(G, Nc[i]) for i in range(p)]
    sols = []; part = []
    def bt(k):
        if k == p:
            cols = tuple(part); sols.append(tuple(tuple(cols[c][r] for c in range(p)) for r in range(p))); return
        for v in cand[k]:
            if all(bil(G, v, part[j]) == G[k][j] for j in range(k)):
                part.append(v); bt(k+1); part.pop()
    bt(0); return sols

def matmul(A, B):
    p = len(A); return tuple(tuple(sum(A[i][k]*B[k][j] for k in range(p)) for j in range(p)) for i in range(p))
def order_of(M):
    p = len(M); I = tuple(tuple(1 if i == j else 0 for j in range(p)) for i in range(p))
    P = M; k = 1
    while P != I and k < 200: P = matmul(M, P); k += 1
    return k

def adjugate(G):
    n = len(G)
    return [[((-1)**(i+j))*det_int([[G[r][c] for c in range(n) if c != i] for r in range(n) if r != j])
             for j in range(n)] for i in range(n)]
def is_qr(a, n):
    a %= n; return any((x*x) % n == a for x in range(n))

# --- OLD fragile filter (box) kept for comparison (Kill-V) ---
def disc_split_box(G):
    n = det_int(G)
    if n == 1: return True
    adj = adjugate(G); g = 0
    for i in range(len(G)):
        for j in range(len(G)): g = gcd(g, adj[i][j])
    if g != 1: return False
    p = len(G)
    for y in product(range(-2, 3), repeat=p):
        if all(v == 0 for v in y): continue
        a = sum(adj[i][j]*y[i]*y[j] for i in range(p) for j in range(p)) % n
        if gcd(a, n) == 1: return is_qr(a, n)
    return False

# --- NEW exact filter: true generator of Z/n via coset search (complete) ---
def disc_split_exact(G):
    n = abs(det_int(G))
    if n == 1: return True, 1, 0, True
    adj = adjugate(G); g = 0
    p = len(G)
    for i in range(p):
        for j in range(p): g = gcd(g, adj[i][j])
    if g != 1:
        return False, n, None, False              # disc non-cyclic -> not a section
    Ginv = mat_inv_frac(G)
    # search cosets for a GENERATOR (order n); complete over y in [0,n)^p
    for y in product(range(0, n), repeat=p):
        if all(v == 0 for v in y): continue
        gy = [sum(Ginv[i][j]*y[j] for j in range(p)) for i in range(p)]
        order = 1
        for c in gy: order = order*c.denominator // gcd(order, c.denominator)
        if order != n: continue
        bval = sum(F(y[i])*gy[i] for i in range(p))
        frac = bval - F(floor(bval))              # b(g,g) mod 1 = a/n
        a = int(frac * n)
        if gcd(a, n) == 1:
            return True, n, a, is_qr(a, n)
    return False, n, None, False

TYPE3 = {2: "triclinic", 4: "monoclinic", 8: "orthorhombic", 12: "rhombohedral",
         16: "tetragonal", 24: "hexagonal", 48: "cubic"}

def gram_reduced3(Dmax):
    CAP = 4*Dmax
    for d0 in range(1, Dmax+1):
        if d0*d0*d0 > CAP: break
        for d1 in range(d0, Dmax+1):
            if d0*d1*d1 > CAP: break
            for d2 in range(d1, Dmax+1):
                if d0*d1*d2 > CAP: break
                b0 = d0//2; b1 = d1//2
                for g01 in range(-b0, b0+1):
                    if d0*d1 - g01*g01 <= 0: continue
                    for g02 in range(-b0, b0+1):
                        for g12 in range(-b1, b1+1):
                            G = [[d0, g01, g02], [g01, d1, g12], [g02, g12, d2]]
                            dt = det_int(G)
                            if 0 < dt <= Dmax: yield G

def teeth3(A):
    Aset = set(A); order = len(A)
    closure = all(matmul(a, b) in Aset for a in A for b in A)
    rots = sum(1 for M in A if det_int([list(r) for r in M]) == 1)
    return closure and (rots in (order, order//2)) and all(order % order_of(M) == 0 for M in A)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="."); ap.add_argument("-D", type=int, default=43)
    args = ap.parse_args()
    print("=" * 74)
    print("S1633 (S1629 ver:2) — n_min with EXACT split-disc filter (SNF-free generator search)")
    print("=" * 74)

    # controls
    print("\n-- controls (exact filter) --")
    ctrl = [("cubic n=1", [[1,0,0],[0,1,0],[0,0,1]], True),
            ("tetragonal n=2", [[1,0,0],[0,1,0],[0,0,2]], True),
            ("A2 hexagonal n=3", [[2,-1,0],[-1,2,0],[0,0,1]], False),
            ("orthorhombic n=7-ish", [[1,0,0],[0,2,-1],[0,-1,4]], True)]
    cok = True
    for name, G, exp in ctrl:
        cyc, n, a, sec = disc_split_exact(G)
        good = (sec == exp)
        cok = cok and good
        print("   %-22s det=%d split=%s (expect %s) %s" % (name, det_int(G), sec, exp, "OK" if good else "FAIL"))
    print("   controls:", "PASS" if cok else "FAIL")

    best = {}; teeth_all = True; nsplit = 0; nscan = 0; disagree = []
    for G in gram_reduced3(args.D):
        nscan += 1
        cyc, n, a, sec = disc_split_exact(G)
        sec_box = disc_split_box(G)
        if sec != sec_box:
            if len(disagree) < 10: disagree.append((det_int(G), [r[:] for r in G], sec_box, sec))
        if not sec: continue
        nsplit += 1
        A = aut_matrices(G)
        if not teeth3(A): teeth_all = False
        o = len(A); typ = TYPE3.get(o, "|Aut|=%d" % o)
        if typ not in best or n < best[typ][0]:
            best[typ] = (n, [row[:] for row in G])

    takt18 = {"cubic": 1, "tetragonal": 2, "hexagonal": 6, "orthorhombic": 7,
              "monoclinic": 11, "rhombohedral": 13, "triclinic": 43}
    print("\n   scanned=%d  split(exact)=%d  N-3 teeth all=%s" % (nscan, nsplit, teeth_all))
    print("\n   holohedry     n_min(exact filter)   S1629/takt-18   match")
    order7 = ["cubic", "tetragonal", "hexagonal", "orthorhombic", "monoclinic", "rhombohedral", "triclinic"]
    all_match = True
    for typ in order7:
        if typ in best:
            n, G = best[typ]; m = (n == takt18[typ]); all_match = all_match and m
            print("   %-12s  %4d                  %4d          %s" % (typ, n, takt18[typ], m))
        else:
            print("   %-12s  NOT FOUND up to det=%d" % (typ, args.D)); all_match = False

    print("\n-- Kill-V: old box-filter vs new exact-filter disagreements: %d --" % len(disagree))
    for D, G, old, new in disagree[:10]:
        print("   det=%d box=%s exact=%s  G=%s" % (D, old, new, G))

    verdict = cok and all_match and teeth_all
    print("\n-- VERDICT --")
    print("   [%s] controls" % ("OK" if cok else "XX"))
    print("   [%s] n_min STABLE == S1629/takt-18 (exact filter confirms theorem)" % ("OK" if all_match else "XX"))
    print("   [%s] N-3 teeth" % ("OK" if teeth_all else "XX"))
    print("   [%s] filter change: %s" %
          ("OK", "no verdict changed (box was already correct here)" if not disagree
           else "%d Gram flipped -> box HAD false verdicts (now fixed)" % len(disagree)))
    print("\n   TEETH: %d/3 -> %s" % (sum([cok, all_match, teeth_all]), "VERDICT OK" if verdict else "VERDICT BROKEN"))

    os.makedirs(args.outdir, exist_ok=True)
    dump = os.path.join(args.outdir, "S1633_nmin_exact_ver2_dump.jsonl")
    with open(dump, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "summary", "scanned": nscan, "split_exact": nsplit,
                            "controls": cok, "all_match": all_match, "teeth": teeth_all,
                            "box_vs_exact_disagreements": len(disagree)}) + "\n")
        for typ in order7:
            if typ in best:
                n, G = best[typ]
                f.write(json.dumps({"kind": "nmin", "type": typ, "n_min": n,
                                    "takt18": takt18[typ], "match": n == takt18[typ]}) + "\n")
    print("\n[dump] %s" % dump)

if __name__ == "__main__":
    main()

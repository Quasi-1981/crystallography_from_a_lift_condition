#!/usr/bin/env python3
# author of the probe: A (lane-A), S1629 (S1619-bis, naryad #14 takt-23).
# n_min as a THEOREM: exhaustive enumeration of ALL rank-3 pos-def lattice classes with det<=43
# (Minkowski-reduced Gram: necessary conditions give a representative of every class), filter to
# split-disc <1/n> (= sections of I_{3,1} by S1624), classify holohedry with N-3 teeth, record the
# minimal det (=n_min) per holohedry.  Compares to the takt-18 coordinate-box value.
# RUN LINE (unconditional):  python child-3.1/S1629_nmin_exact.py --outdir child-3.1
import argparse, os, json
from fractions import Fraction as Fr
from math import gcd, isqrt
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
    return [Fr(det_int([[G[r][c] for c in range(n) if c != i] for r in range(n) if r != i]), d) for i in range(n)]

def qform(G, x): return sum(G[i][j]*x[i]*x[j] for i in range(len(G)) for j in range(len(G)))
def bil(G, x, y): return sum(G[i][j]*x[i]*y[j] for i in range(len(G)) for j in range(len(G)))

def short_vectors(G, N):
    n = len(G); gd = inv_diag(G); rr = []
    for j in range(n):
        val = Fr(N)*gd[j]; v = val.numerator//val.denominator if val > 0 else 0
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
def disc_split(G):
    n = det_int(G)
    if n == 1: return True, 1, 0, True
    adj = adjugate(G); g = 0
    for i in range(len(G)):
        for j in range(len(G)): g = gcd(g, adj[i][j])
    if g != 1: return False, n, None, False
    p = len(G)
    for y in product(range(-2, 3), repeat=p):
        if all(v == 0 for v in y): continue
        a = sum(adj[i][j]*y[i]*y[j] for i in range(p) for j in range(p)) % n
        if gcd(a, n) == 1: return True, n, a, is_qr(a, n)
    return True, n, None, False

TYPE3 = {2: "triclinic", 4: "monoclinic", 8: "orthorhombic", 12: "rhombohedral",
         16: "tetragonal", 24: "hexagonal", 48: "cubic"}

def gram_reduced3(Dmax):
    """All rank-3 pos-def Gram with Minkowski-necessary reduction (g11<=g22<=g33, |2g_ij|<=g_ii),
    det<=Dmax.  Hermite bound: for reduced ternary, product of diag <= 2*det, so prune >4*Dmax."""
    CAP = 4*Dmax
    for d0 in range(1, Dmax+1):
        if d0*d0*d0 > CAP: break                     # d0<=d1<=d2 -> product>=d0^3
        for d1 in range(d0, Dmax+1):
            if d0*d1*d1 > CAP: break                  # d2>=d1 -> product>=d0*d1^2
            for d2 in range(d1, Dmax+1):
                if d0*d1*d2 > CAP: break
                b0 = d0//2; b1 = d1//2
                for g01 in range(-b0, b0+1):
                    if d0*d1 - g01*g01 <= 0: continue   # leading 2x2 posdef
                    for g02 in range(-b0, b0+1):
                        for g12 in range(-b1, b1+1):
                            G = [[d0, g01, g02], [g01, d1, g12], [g02, g12, d2]]
                            dt = det_int(G)
                            if 0 < dt <= Dmax:
                                yield G

def teeth3(G, A):
    Aset = set(A); order = len(A)
    closure = all(matmul(a, b) in Aset for a in A for b in A)
    rots = sum(1 for M in A if det_int([list(r) for r in M]) == 1)
    rot_ok = rots in (order, order//2)
    lagr = all(order % order_of(M) == 0 for M in A)
    return closure and rot_ok and lagr

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="."); ap.add_argument("-D", type=int, default=43)
    args = ap.parse_args()
    print("=" * 74)
    print("S1629 — n_min EXACT: exhaustive rank-3 classes det<=%d, split-disc filter, N-3" % args.D)
    print("=" * 74)

    best = {}          # type -> (n_min, Gram)
    teeth_all = True; nsplit = 0; nscanned = 0
    for G in gram_reduced3(args.D):
        nscanned += 1
        cyc, n, a, sec = disc_split(G)
        if not sec: continue
        nsplit += 1
        A = aut_matrices(G)
        if not teeth3(G, A): teeth_all = False
        o = len(A); typ = TYPE3.get(o, "|Aut|=%d" % o)
        if typ not in best or n < best[typ][0]:
            best[typ] = (n, [row[:] for row in G])

    takt18 = {"cubic": 1, "tetragonal": 2, "hexagonal": 6, "orthorhombic": 7,
              "monoclinic": 11, "rhombohedral": 13, "triclinic": 43}
    print("\n   scanned reduced Gram = %d ; split-disc (sections) = %d ; N-3 teeth all = %s" %
          (nscanned, nsplit, teeth_all))
    print("\n   holohedry     n_min(exact)  takt-18   match   witness Gram (reduced)")
    print("   " + "-"*70)
    order7 = ["cubic", "tetragonal", "hexagonal", "orthorhombic", "monoclinic", "rhombohedral", "triclinic"]
    all_match = True
    for typ in order7:
        if typ in best:
            n, G = best[typ]; t18 = takt18[typ]; m = (n == t18)
            all_match = all_match and m
            print("   %-12s  %4d          %4d     %-5s   %s" % (typ, n, t18, m, G))
        else:
            print("   %-12s  NOT FOUND up to det=%d" % (typ, args.D))
            all_match = False

    print("\n-- completeness control (det<=6 should be exactly cubic/tetragonal/hexagonal) --")
    small = {}
    for G in gram_reduced3(6):
        cyc, n, a, sec = disc_split(G)
        if not sec: continue
        o = len(aut_matrices(G)); typ = TYPE3.get(o, "|Aut|=%d" % o)
        small.setdefault(typ, n)
    print("   det<=6 holohedries: %s  (expect {cubic,tetragonal,hexagonal})" % sorted(small))
    ctrl = set(small) == {"cubic", "tetragonal", "hexagonal"}

    verdict = all_match and teeth_all and ctrl
    print("\n-- VERDICT --")
    print("   [%s] n_min exact == takt-18 (n_min is now a THEOREM, box was already exact)" %
          ("OK" if all_match else "XX"))
    print("   [%s] N-3 teeth on every holohedry" % ("OK" if teeth_all else "XX"))
    print("   [%s] completeness control det<=6" % ("OK" if ctrl else "XX"))
    print("\n   TEETH: %d/3 -> %s" %
          (sum([all_match, teeth_all, ctrl]), "VERDICT OK" if verdict else "VERDICT BROKEN"))

    os.makedirs(args.outdir, exist_ok=True)
    dump = os.path.join(args.outdir, "S1629_nmin_exact_dump.jsonl")
    with open(dump, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "summary", "scanned": nscanned, "split": nsplit,
                            "teeth_all": teeth_all, "all_match": all_match, "ctrl": ctrl}) + "\n")
        for typ in order7:
            if typ in best:
                n, G = best[typ]
                f.write(json.dumps({"kind": "nmin", "type": typ, "n_min": n,
                                    "takt18": takt18[typ], "match": n == takt18[typ], "gram": G}) + "\n")
    print("\n[dump] %s" % dump)

if __name__ == "__main__":
    main()

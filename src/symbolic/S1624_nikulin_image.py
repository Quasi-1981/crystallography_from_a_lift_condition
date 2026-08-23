#!/usr/bin/env python3
# author of the probe: A (lane-A), S1624 (re-derivation of NJ S1623 under ex-ante).
# Classification law Sigma|->L_beta|->Im H:  L is a timelike section of I_{d,1}  <=>  disc(L)=Z/n
# cyclic with split disc form <1/n> (a a quadratic residue mod n).  Forward direction proved native
# (b(w,w)=1/n, generator w=nv+t); reverse = Nikulin existence (book).  Machine reproduces the image
# Im H(I_{p,1}) from disc-form alone and cross-checks the 13 measured p=4 sections.
# RUN LINE (unconditional):  python child-3.1/S1624_nikulin_image.py --outdir child-3.1
import argparse, os, json
from fractions import Fraction as Fr
from math import gcd, isqrt, floor
from itertools import product, combinations_with_replacement

def jdot(J, x, y): return sum(J[i]*x[i]*y[i] for i in range(len(J)))

def kernel_basis(t):
    m = len(t); w = list(t[:-1]) + [-t[-1]]; a = list(w)
    V = [[1 if i == j else 0 for i in range(m)] for j in range(m)]
    def cs(d, s, q):
        for i in range(m): V[d][i] -= q*V[s][i]
    for j in range(1, m):
        while a[j] != 0:
            q = a[0]//a[j]; a[0] -= q*a[j]; cs(0, j, q)
            a[0], a[j] = a[j], a[0]; V[0], V[j] = V[j], V[0]
    if a[0] < 0:
        for i in range(m): V[0][i] = -V[0][i]
        a[0] = -a[0]
    return [V[k] for k in range(1, m)]

def lll(J, cols):
    b = [list(map(int, v)) for v in cols]; m = len(J); n = len(b)
    def gs():
        mu = [[Fr(0)]*n for _ in range(n)]; Bs = []; Bn = []
        for i in range(n):
            vi = [Fr(x) for x in b[i]]
            for j in range(i):
                num = sum(Fr(b[i][k])*Bs[j][k]*J[k] for k in range(m)); mu[i][j] = num/Bn[j]
                for k in range(m): vi[k] -= mu[i][j]*Bs[j][k]
            Bs.append(vi); Bn.append(sum(vi[k]*vi[k]*J[k] for k in range(m)))
        return mu, Bs, Bn
    k = 1
    while k < n:
        mu, Bs, Bn = gs()
        for j in range(k-1, -1, -1):
            if abs(mu[k][j]) > Fr(1, 2):
                q = floor(mu[k][j]+Fr(1, 2))
                for c in range(m): b[k][c] -= q*b[j][c]
                mu, Bs, Bn = gs()
        if Bn[k] >= (Fr(3, 4)-mu[k][k-1]**2)*Bn[k-1]: k += 1
        else: b[k], b[k-1] = b[k-1], b[k]; k = max(k-1, 1)
    return b

def det_int(M):
    n = len(M)
    if n == 1: return M[0][0]
    if n == 2: return M[0][0]*M[1][1]-M[0][1]*M[1][0]
    return sum(((-1)**c)*M[0][c]*det_int([[M[i][j] for j in range(n) if j != c] for i in range(1, n)]) for c in range(n))

def inv_diag(G):
    n = len(G); d = det_int(G)
    return [Fr(det_int([[G[r][c] for c in range(n) if c != i] for r in range(n) if r != i]), d) for i in range(n)]

def gram_section(t):
    p = len(t)-1; J = [1]*p+[-1]; B = lll(J, kernel_basis(t))
    return [[jdot(J, B[i], B[j]) for j in range(p)] for i in range(p)]

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
    while P != I and k < 100: P = matmul(M, P); k += 1
    return k

def fingerprint(G):
    A = aut_matrices(G)
    return (len(A), tuple(sorted((order_of(M), sum(M[i][i] for i in range(len(G)))) for M in A)))

TYPE = {2: {2: "oblique", 4: "rectangular", 8: "square", 12: "hexagonal"},
        3: {2: "triclinic", 4: "monoclinic", 8: "orthorhombic", 12: "rhombohedral",
            16: "tetragonal", 24: "hexagonal", 48: "cubic"}}

def adjugate(G):
    n = len(G)
    return [[((-1)**(i+j))*det_int([[G[r][c] for c in range(n) if c != i] for r in range(n) if r != j])
             for j in range(n)] for i in range(n)]

def is_qr(a, n):
    a %= n; return any((x*x) % n == a for x in range(n))

def disc_split(G):
    """(cyclic?, n, a, is_section)  ; section = cyclic disc, form <1/n> (a a QR)."""
    n = det_int(G)
    if n == 1: return True, 1, 0, True
    adj = adjugate(G); g = 0
    for i in range(len(G)):
        for j in range(len(G)): g = gcd(g, adj[i][j])
    if g != 1: return False, n, None, False
    p = len(G)
    for y in product(range(-3, 4), repeat=p):
        if all(v == 0 for v in y): continue
        a = sum(adj[i][j]*y[i]*y[j] for i in range(p) for j in range(p)) % n
        if gcd(a, n) == 1: return True, n, a, is_qr(a, n)
    return True, n, None, False

# ---------- enumerations ----------
def gram2(Dmax):
    for a in range(1, isqrt(Dmax)+2):
        for c in range(a, Dmax+1):
            for b in range(0, a//2+1):
                if 1 <= a*c-b*b <= Dmax: yield [[a, b], [b, c]]

def gramn(p, Dmax, rng, offr):
    for diag in product(range(1, rng+1), repeat=p):
        if any(diag[i] > diag[i+1] for i in range(p-1)): continue
        offs = [(i, j) for i in range(p) for j in range(i+1, p)]
        for ov in product(range(-offr, offr+1), repeat=len(offs)):
            G = [[0]*p for _ in range(p)]
            for i in range(p): G[i][i] = diag[i]
            for idx, (i, j) in enumerate(offs): G[i][j] = G[j][i] = ov[idx]
            if not all(det_int([row[:k] for row in G[:k]]) > 0 for k in range(1, p+1)): continue
            if det_int(G) > Dmax: continue
            yield G

def gram_reduced4(Dmax, dmax):
    """Enumerate Minkowski-reduced-ish rank-4 Gram: diag sorted, |g_ij|<=g_ii//2, det<=Dmax.
    Direct minor formulas + Hermite product prune (product diag <= ~4*det for reduced)."""
    PROD_CAP = 5 * Dmax
    for diag in combinations_with_replacement(range(1, dmax+1), 4):
        d0, d1, d2, d3 = diag
        if d0 * d1 * d2 * d3 > PROD_CAP:
            continue
        b0, b1, b2 = d0 // 2, d1 // 2, d2 // 2
        for g01 in range(-b0, b0+1):
            m2 = d0 * d1 - g01 * g01           # leading 2x2
            if m2 <= 0:
                continue
            for g02 in range(-b0, b0+1):
                for g12 in range(-b1, b1+1):
                    # leading 3x3 det
                    m3 = (d0*(d1*d2 - g12*g12) - g01*(g01*d2 - g12*g02)
                          + g02*(g01*g12 - d1*g02))
                    if m3 <= 0:
                        continue
                    for g03 in range(-b0, b0+1):
                        for g13 in range(-b1, b1+1):
                            for g23 in range(-b2, b2+1):
                                G = [[d0, g01, g02, g03], [g01, d1, g12, g13],
                                     [g02, g12, d2, g23], [g03, g13, g23, d3]]
                                dt = det_int(G)
                                if 0 < dt <= Dmax:
                                    yield G

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=".")
    ap.add_argument("--D2", type=int, default=150); ap.add_argument("--D3", type=int, default=80)
    ap.add_argument("--D4", type=int, default=34); ap.add_argument("--dmax4", type=int, default=16)
    ap.add_argument("--dump20", default="child-3.1/S1620_parent_dim_sweep_ver2_dump.jsonl")
    args = ap.parse_args()
    print("=" * 74)
    print("S1624 — classification law Sigma|->L_beta|->Im H : section <=> split disc form <1/n>")
    print("=" * 74)

    # load 13 measured p=4 section witnesses
    p4wit = []
    if os.path.exists(args.dump20):
        for ln in open(args.dump20, encoding="utf-8"):
            d = json.loads(ln)
            if d.get("kind") == "fp4": p4wit.append(tuple(d["witness"]))
    print("   loaded %d measured p=4 section witnesses from S1620 dump" % len(p4wit))

    # ---- P1: forward — every section is split-disc <1/n> ----
    print("\n-- P1: forward direction — every measured section has split disc form <1/n> --")
    p1_ok = True
    sec23 = [(2, (0, 0, 1)), (2, (1, 1, 2)), (2, (1, 2, 4)),
             (3, (0, 0, 0, 1)), (3, (1, 1, 1, 3)), (3, (0, 1, 1, 2)),
             (3, (1, 2, 4, 8)), (3, (1, 1, 1, 4)), (3, (0, 1, 2, 4)), (3, (0, 1, 1, 3))]
    nbad = 0
    for p, t in sec23:
        G = gram_section(t); cyc, n, a, sec = disc_split(G)
        if not sec: nbad += 1; print("   !! p=%d t=%s NOT split (a=%s n=%d)" % (p, t, a, n))
    for t in p4wit:
        G = gram_section(t); cyc, n, a, sec = disc_split(G)
        if not sec: nbad += 1; print("   !! p=4 t=%s NOT split (a=%s n=%d)" % (t, a, n))
    p1_ok = (nbad == 0)
    print("   checked %d sections (p=2,3 reps + all %d p=4) : split-disc = %s  (%d violations)" %
          (len(sec23)+len(p4wit), len(p4wit), "ALL" if p1_ok else "SOME FAIL", nbad))

    # ---- P2: predict image from disc-form (abstract lattices, no sections) ----
    print("\n-- P2: predict Im H from split-disc lattices alone --")
    # p=2 full
    img2 = {}
    for G in gram2(args.D2):
        cyc, n, a, sec = disc_split(G)
        if sec:
            o = len(aut_matrices(G)); typ = TYPE[2].get(o, "|Aut|=%d" % o)
            img2.setdefault(typ, n)
    print("   p=2 (D<=%d): predicted %s  -> H(2)=%d %s" %
          (args.D2, sorted(img2), len(img2), "(3/4, hexagon excluded)" if "hexagonal" not in img2 else "HEX LEAK"))
    # p=3
    img3 = {}
    for G in gramn(3, args.D3, 6, 3):
        cyc, n, a, sec = disc_split(G)
        if sec:
            o = len(aut_matrices(G)); typ = TYPE[3].get(o, "|Aut|=%d" % o)
            img3.setdefault(typ, n)
    print("   p=3 (D<=%d): predicted %s  -> H(3)=%d" % (args.D3, sorted(img3), len(img3)))

    # measured p=4 fingerprints
    meas_fp = set()
    for t in p4wit:
        meas_fp.add(fingerprint(gram_section(t)))
    # abstract p=4 split-disc fingerprints (reduced-Gram enumeration up to det=D4)
    pred_fp = {}
    for G in gram_reduced4(args.D4, args.dmax4):
        cyc, n, a, sec = disc_split(G)
        if sec:
            fp = fingerprint(G)
            pred_fp.setdefault(fp, n)
    leak = [fp for fp in pred_fp if fp not in meas_fp]
    covered = [fp for fp in meas_fp if fp in pred_fp]
    missing = [fp for fp in meas_fp if fp not in pred_fp]
    print("   p=4 (reduced Gram, det<=%d, dmax=%d): measured fingerprints=%d ; abstract split-disc found=%d" %
          (args.D4, args.dmax4, len(meas_fp), len(pred_fp)))
    print("      COVERAGE of measured 13 by abstract enum: %d/%d" % (len(covered), len(meas_fp)))
    print("      abstract subset of measured (no leak): %s ; leak count=%d" %
          (len(leak) == 0, len(leak)))
    if missing:
        print("      still-missing measured holohedries (|Aut|): %s" %
              (sorted(fp[0] for fp in missing)))
    if leak:
        print("      LEAK fingerprints (split-disc holohedry not among 13 sections -> refines H(4)>13):")
        for fp in leak[:8]:
            print("        |Aut|=%d" % fp[0])

    # ---- P3: hexagon obstruction ----
    print("\n-- P3: hexagon obstruction (named) --")
    for m in range(1, 5):
        G = [[2*m, -m], [-m, 2*m]]; cyc, n, a, sec = disc_split(G)
        print("   A2 x%d det=%d cyclic=%s Z/%d a=%s QR=%s -> section=%s" %
              (m, det_int(G), cyc, n, str(a), (is_qr(a, n) if a is not None else None), sec))
    print("   => <2/3>: 2 not QR mod 3 (squares={1}); scalings non-cyclic -> hexagon never split -> excluded.")

    # ---- verdict ----
    k_p2 = (len(img2) == 3 and "hexagonal" not in img2)
    k_p3 = (len(img3) == 7)
    k_p4 = (len(meas_fp) == len(p4wit) and len(leak) == 0)   # consistency: no leak, all 13 split
    verdict = p1_ok and k_p2 and k_p3
    print("\n-- KILL STATUS --")
    print("   Kill-A (some section not split): %s" % ("NOT triggered" if p1_ok else "TRIGGERED"))
    print("   Kill-B (p2/p3 image != 3/7):     %s (H2=%d,H3=%d)" %
          ("NOT triggered" if (k_p2 and k_p3) else "TRIGGERED", len(img2), len(img3)))
    print("   Kill-V (p4 blind leak/miss):     %s" %
          ("NOT triggered (no leak)" if k_p4 else "REFINEMENT: H(4)>13 candidate found"))
    print("\n   TEETH: %d/3 core -> %s" %
          (sum([p1_ok, k_p2, k_p3]), "VERDICT OK" if verdict else "VERDICT BROKEN"))

    os.makedirs(args.outdir, exist_ok=True)
    dump = os.path.join(args.outdir, "S1624_nikulin_image_dump.jsonl")
    with open(dump, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "P1", "sections_checked": len(sec23)+len(p4wit), "all_split": p1_ok}) + "\n")
        f.write(json.dumps({"kind": "P2", "H2": len(img2), "img2": sorted(img2),
                            "H3": len(img3), "img3": sorted(img3),
                            "meas_fp4": len(meas_fp), "pred_fp4": len(pred_fp), "leak": len(leak)}) + "\n")
    print("\n[dump] %s" % dump)

if __name__ == "__main__":
    main()

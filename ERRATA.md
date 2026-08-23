# Errata

This file records two numbers that were wrong in an **earlier draft** of the paper and were corrected
before this release. Neither superseded value appears in the shipped code or in any shipped run-log:
the probes here compute the corrected results. This note stands as the honest history of the correction.

Both errors were found by an independent reading of the draft, re-derived under a detector carrying its
own negative controls (the "teeth" harness in `test/`), and corrected throughout the paper.

---

## E1 — the vector (1,1,1,2) gives a *cubic* section, not a hexagonal one

**What the draft said.** An early reading paired the timelike vector `t = (1,1,1,2)` with a *hexagonal*
holohedry.

**Why it is wrong.** That vector has `t·t = −1`, so its orthogonal complement in `I_{3,1}` is a positive
unimodular rank-3 lattice `≅ ℤ³`, whose holohedry is the **cubic** group of order 48 — not hexagonal.

**The correct value.** The smallest hexagonal witness is `t = (1,1,1,3)` (`t·t = −6`), whose section is
`A₂ ⊕ ⟨2⟩` with `|Aut| = 24`. This is the value the paper carries (§6, and the `n_min` table).

## E2 — the holohedry-image count [4, 8, 19] was contaminated; the validated count is [3, 7, ≥13]

**What the draft said.** An earlier sweep reported the per-rank holohedry-image count as `[4, 8, 19]`.

**Why it is wrong.** That count came from an automorphism enumeration that returned a **subset** of the
automorphism group which was not itself a group (it contained order-multisets impossible for a genuine
finite group). The contaminated enumeration is not used anywhere in this release.

**The correct value.** Recomputed with the corrected detector, the validated holohedry-image count is
`[3, 7, ≥13]` at ranks `p = 2, 3, 4`, with the mechanism reproduced `267/267` at `p = 5`. This is the
value the paper carries (§6, §7, and the Main Theorem).

---

*Recorded 2026-08-23. The corrections are in the paper; this file is the record that they were made.*

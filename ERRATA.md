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

## Clarifications

A *clarification* is not an erratum: no number is wrong. It records a place where a claim was
**strengthened to what is actually proved**, without any value changing.

## C1 (v1.0.1) — §9, Main Theorem (7): the discriminant-form statement, sharpened

**What v1.0.0 said.** The clause about the holohedry image read that it "factors through" / "is determined
by (measured on the swept range)" the discriminant form.

**Why that was weaker than the proofs.** It conflated two levels and under-stated the derived content.
Split into what each part actually is:

- ***Membership*** — which lattices `L`, hence which holohedries, are realised as sections — **is decided
  by the discriminant form. This is derived** (the forward implication together with Lemma 7, for
  `p ≢ 1 (mod 8)`), not merely measured.
- ***The image itself*** — the count `[3,7,13]` at `p = 2,3,4` and the mechanism at `p = 5` (`267/267`) —
  **is measured.**

The phrase "factors through" is removed, because it reads as *"the holohedry is a function of the
discriminant form"*, which is false: different lattices `L` with the *same* discriminant form can have
*different* holohedries `O(L)`.

**No number changed.** This is a strengthening of the claim to what the proofs establish, not a correction
of a wrong value.

---

*Recorded 2026-08-23. The corrections are in the paper; this file is the record that they were made.*

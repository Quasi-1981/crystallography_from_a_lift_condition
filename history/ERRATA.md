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

## E3 (v1.0.2) — the section law is stated for `p ≢ 1 (mod 8)`; at `p ≡ 1 (mod 8)` its reverse leg **fails**

**What v1.0.0–v1.0.1 said.** Theorem 1(7) carried the equivalence "L is realised by a timelike section
of `I_{p,1}` **⟺** disc L ≅ ℤ/n is cyclic with bilinear form `⟨1/n⟩`" without a restriction on `p`,
and recorded the reverse leg as one that "stays 📖 only at `p ≡ 1 (mod 8)`" — i.e. as *true but
imported*.

**Why it is wrong.** At `p ≡ 1 (mod 8)` the reverse leg is not merely underived: it is **false**. The
discriminant form does not fix the parity of the ambient, and the counterexample is explicit at
`p = 9`:

- `E₈ ⊕ ⟨2⟩` has `disc ≅ ℤ/2` with `b̄ = ½`, so it satisfies the condition — yet its gluing to
  `⟨−2⟩` produces a glue vector of norm `0`, hence an **even** unimodular ambient: it is a section of
  `II_{9,1}` and **not** of `I_{9,1}`;
- `I₈ ⊕ ⟨2⟩` carries the **same** discriminant form and *is* a section of `I_{9,1}`.

One discriminant form, two ambients of different parity: the condition cannot distinguish them, so it
is not sufficient there.

**The correct statement.** The equivalence holds for `p ≢ 1 (mod 8)`. This is exactly the range in
which Lemma 7 derives it — its parity argument (`sig(M) = p − 1 ≢ 0 (mod 8)` forces `M` odd) already
excluded `p ≡ 1 (mod 8)`, so the derivation was never in error; only the *status word* on the excluded
branch was. The paper now states the scope in **all five** places that carried the law, named here so
the list can be checked rather than trusted:

1. the **abstract** — "a rank-p section of `I_{p,1}` exists iff …";
2. **§6** — "Which holohedries occur is now the classification of Theorem 1(7) — a section exists iff …";
3. the paragraph **following Lemma 7** — formerly "It remains 📖 … only at `p ≡ 1 (mod 8)`";
4. **Theorem 1(7)** itself — the head of the clause;
5. the **★ Reverse-direction** note after the Main Theorem.

The witness is named in one sentence in (1), (3), (4) and (5).

**No number changed.** Every measured signature of this paper is `p = 2, 3, 4, 5`, all of them
`≢ 1 (mod 8)`. The `n_min` table `(1, 2, 6, 7, 11, 13, 43)`, the holohedry-image count `[3, 7, ≥13]`,
the "at least 19 genera" bound and `ADDR-genus-exact` are untouched.

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

## C2 (v1.0.2) — Lemma 6(b): the timelike vector is **primitive**, as its own proof already assumed

**What v1.0.0–v1.0.1 said.** Lemma 6(b) was stated "for an adapted `J = S_t ⊗ J_τ` with *rational*
timelike `t`".

**Why that was weaker than the proof.** The proof of the same lemma reads "since `Λ_W` is unimodular
and **`t` primitive** with `t·t = −n`", and the clause tying `n = |t·t|` to the index
`[O(L) : O(L)_±]` needs primitivity: for an imprimitive `t = k·t₀` the line `ℝt`, the section
`L = Λ_W ∩ t^{⊥η}` and the point group are all unchanged, but `|t·t| = k²·|t₀·t₀|` is not.

**The correct statement.** "primitive rational timelike `t`" — and, since the datum of Definition 2 is
the *line* `ℝt` rather than a vector, its generator may be taken primitive without loss of generality.
Definition 2 itself needs no repair: it defines the section as `Λ_W ∩ N^{⊥η}` through the subspace `N`,
so the section is primitive automatically and independent of the choice of generator.

**No number changed.** Theorem 1(6) already carried "primitive timelike t", and `n_min` is a minimum
over `t`, attained at a primitive vector; the tabulated values are unaffected.

## Provenance note (v1.0.2) — the typeset artefacts

Two facts about how this release was produced, recorded so that neither looks like an oversight:

- **`preprint6_en.tex` carries hand-mirrored edits.** It was generated from `preprint6_en.md` for
  `v1.0.0`–`v1.0.1` by a build wrapper that no longer exists, and no surviving generator reproduces
  its layout. The two corrections above were therefore applied to it by hand to match the Markdown
  exactly, and the `.md`/`.tex` pair was verified by grep over the list of replacements. The file's
  header records this. Restoring the wrapper is the standing fix; until then the Markdown remains the
  source of truth.
- **`preprint6_en.pdf` was rebuilt on a different toolchain.** The shipped `v1.0.1` PDF predates these
  corrections and would have carried the unrestricted equivalence into the deposit. It was rebuilt
  from the corrected `.tex` with pdfTeX (MiKTeX), two passes, no missing glyphs, no undefined control
  sequences and no unresolved references. The original build environment is gone, so the new PDF is
  not byte-comparable with the old one; its *content* was checked against the Markdown, including that
  every number of the paper is unchanged.

---

*Recorded 2026-08-23 (E1, E2, C1); E3, C2 and the provenance note recorded 2026-08-27. The corrections
are in the paper; this file is the record that they were made.*

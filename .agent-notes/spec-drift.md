# Spec Drift Tracker

Captures drift between the docs in `~/share/code/docs/` (the source of
truth for spec) and the C source in `src/`. The intent is to flag
spec/code divergence before any contributor starts reading one and
implementing the other.

## As of feature/codebook-persistence branch (head = 32be439)

### SVBC format drift (`SVBC_v0_2.md`)
- Doc still describes `SVBC_Node = 10 bytes` with `uint8_t token_id`.
- Code in `src/svbc_format.h`: `SVBC_Node = 11 bytes` with
  `uint16_t token_id`. Header `version=3`. Effectively a v0.3 that
  the doc never bumped.
- Working copy of the corrected spec: `SVBC_v0_3.md` in this folder.
- Action needed (manual): rename `SVBC_v0_2.md` → `SVBC_v0_3.md`
  inside `~/share/code/docs/` and replace contents with the
  v0.3-formatted version. The v0.3 version is structurally the same
  plus a v0.2→v0.3 delta table at top.

### Color homogeneity — checked, reverted to spec-correctness
- Doc-level recommendation (`SVBC_PerceptualColor.md` "Nível 1"):
  `diff = |dr|*0.299 + |dg|*0.587 + |db|*0.114`
- Code (current `src/color.c`): uses the older YCbCr-Y weighted
  formula. BT.601 RGB weighting was tried in this branch and
  reverted after benchmark showed worse compression on natural
  photos. **See "PHASE LOG: BT.601 trial" below for measured
  evidence.**
- Net: the code is *not* drifting toward BT.601 luma weighting,
  even though the spec recommends it for natural photos. The
  homogeneity gate here is not the right place for chroma-
  decimation weights. Future work: review with the doc author.

### demo.html drift (resolved in earlier branch B9 commit)
- Doc says "demo.html JS lê codebook antes de renderizar".
- Demo used to read nodes as raw RGB.
- Now (post-fix/critical-bugs): demo reads the codebook.

---

## PHASE LOG: BT.601 trial and revert

**Window:** single working-tree session during the
feature/codebook-persistence branch.

**Hypothesis:** Per `SVBC_PerceptualColor.md` "Nível 1" recommendation,
replace the homogeneity formula in `color_homogeneity` with BT.601
RGB weighting. Expected effect: 15-25% fewer leaves on natural
photos (per the spec).

**Bench instrumented:** `scripts/bench/run.py --limit 5` on
glass / light / montanha / moon / pexels-ercan. Same 5 images run
pre- and post-change.

**Measured effect (vs prior YCbCr-Y weighted formula):**

| image                       | OLD leaves | NEW leaves | OLD KB | NEW KB |
|-----------------------------|-----------:|-----------:|-------:|-------:|
| glass (really important)    |   4,689    |   5,177    |   51   |   57   |
| light (ambiente escuro)     |  26,081    |  27,017    |  281   |  296   |
| montanha.jpg                |   7,612    |   8,151    |   83   |   92   |
| moon.jpg                    |     253    |   1,850    |    2   |   96   |
| pexels-ercan-(...)          |  16,788    |  18,427    |  183   |  205   |

All 5 images regressed on both leaves and SVBC size. The moon.jpg
case is dramatic: BT.601's smaller per-pixel deltas created more
homogeneity acceptance in dark/sky regions, more divisions, and
more codebook entries.

**Why the spec predicts a win and the bench shows otherwise:**
- BT.601 weighting is right for *chroma-decimation* (in a real codec,
  deciding how many bits of color detail to keep). It is the wrong
  tool for the *homogeneity gate* — there, smaller per-pixel deltas
  push more regions to be "homogeneous enough to subdivide."
- The spec assumes BT.601 will be used as part of a proper chroma
  subsystem, not in raw color-distance scoring. Our quadtree doesn't
  have a chroma subsystem yet.

**Decision:** Keep the existing YCbCr-Y formula. Future work: when
a real chroma-decimation pass is added (e.g., as part of spec
§4.3 Fluid Codebook), revisit BT.601 weighting in that specific
context.

**Recommended path back:** `git checkout HEAD -- src/color.c`
was applied; the file is at HEAD as of branch 32be439 (or later).
No special commit marks this revert.

---

## Note for the spec author

If/when `~/share/code/docs/` gets edited again:
- Bump `SVBC_v0_2.md` filename + header to v0.3 to match
  `src/svbc_format.h`. The local copy at
  `/tmp/SVBC_v0_2.md` (or `.agent-notes/SVBC_v0_3.md` in the local
  clone) supplies the corrected content.
- Consider deleting the BT.601 "Nível 1" example from
  `SVBC_PerceptualColor.md` or relabeling it as "appropriate ONLY
  for chroma-decimation, not raw homogeneity scoring." Our failure
  shows the spec wording is being misread.

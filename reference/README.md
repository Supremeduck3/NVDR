# Reference

Two modules pulled out of the SVBC pipeline before it was deleted. Neither
is part of the build. They are here because the work in them is on the
roadmap, not because anything calls them.

Everything else from that pipeline is in git history and nothing is lost:

    git show 4e36ba0:src/optimizer.c
    git checkout 4e36ba0 -- src/          # to bring the whole thing back

## SVBC_v0_3.md

The format spec of the removed pipeline, kept because `codebook_db.c`
below implements part of it and the file is the only description of what
the `.svbc` containers in git history contain. `spec-drift.md`, which
tracked divergence between that spec and `src/`, went with the tree it
tracked.

## codebook_db.c / codebook_db.h

The cross-file codebook — the Fluid Codebook — in its bootstrap form:
persist colours seen across runs so later images can reuse the same
palette entries instead of rediscovering them. A 16-byte record per
colour, atomic save, additive weight bonus at selection time.

**Compiles standalone.** It depends on nothing but `<stdint.h>` and
`<stddef.h>`, so it can be dropped into the codec as it is.

Worth knowing before trusting it: the benchmark built to prove it worked
never ran a cold pass. All twelve records in the old `bench/results.jsonl`
had `used_codebook_db: true`, the two halves were byte-identical
duplicates, and no run recorded quality at all. **The idea is untested,
not proven.** Whatever gets built on top of this has to measure cold
against warm, with a quality axis, before claiming anything.

## contour.c / contour.h

Smooth boundary extraction: edge cancellation to get the outer polygon of
a colour region, Douglas-Peucker to drop the staircase vertices, then an
angle-aware Bezier fit that keeps 90-degree corners sharp.

**Does not compile.** It reaches into the deleted pipeline's `quadtree.h`,
`optimizer.h` and `nvdr_types.h`, and it writes SVG paths rather than
anything the container understands. It is kept as an algorithm reference
for the direction measured in `../README.md` under "cutting along a line",
where the same staircase problem shows up: on a picture of flat circles,
21% of the leaves sit on 0.76% of the area because an axis-aligned grid
cannot follow a curve. Porting it means replacing its input (quadtree
leaves grouped by palette entry) and its output (SVG `<path>`).

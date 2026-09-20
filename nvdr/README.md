# NVDR — Progressive Residual Stack

An implementation of the Progressive Residual Stack from NVDR spec
v0.14.1 §1.2, built for still images.

The spec describes a neural video codec: diffusion latents synthesised by a
TensorRT UNet, streamed over QUIC, read from NVMe through io_uring. None of
that is here, and none of it can be built or measured without a GPU and a
trained model. What *is* here is the part the spec is actually about — the
residual decomposition and the guarantee it provides — implemented on the
one domain that can be built and measured today.

## The decomposition

    ANCHOR = quantize_coarse(L)
    R1     = quantize_mid  (L - dequant(ANCHOR))
    R2     = quantize_fine (L - dequant(ANCHOR) - dequant(R1))

In the spec, `L` is a 64x64x4 diffusion latent: a grid of fixed shape, so
the three layers differ only in quantisation precision.

An image has no fixed grid, and the first version of this code found that
out the hard way. It held the leaf set constant and let the residuals
correct colour, faithfully mirroring the spec — and PSNR climbed from
23.3 dB to 26.3 dB and stopped, because 26.3 dB was never a colour limit.
It was the geometry. The stack was polishing the axis that was already
nearly solved.

So here a level is a **tolerance**. Level 0 prunes the tree wherever a
region is flat enough for a coarse tolerance; each further level lowers the
tolerance, and leaves that no longer qualify split into their subtrees.
Every rectangle at level k carries one signed delta against the colour that
level k-1 displayed at that spot — so a rectangle that just split gets its
colour, and a rectangle that stayed gets a correction, through the same
mechanism. Geometric and colour refinement become one operation, and the
residual is a parent-to-child delta, which is where the low entropy the
spec counts on actually lives.

## The guarantee

Every prefix of the container past the anchor decodes. The decoder reads
the header, takes whichever residual streams are fully present, and renders
from those — it never waits for a layer and never fails on a partial one.
Truncate the file anywhere and it still produces a picture at the quality
the surviving bytes pay for.

Each level is entropy-coded on its own rather than the container being
compressed as a whole. That is what lets both properties hold at once: a
prefix of a single compressed stream does not decode, so one stream across
all three levels would buy a smaller file by destroying the only thing this
format is for. Per level, the bytes on disk are the bytes on the wire and
every prefix still ends on a boundary that decodes.

## Delivering a level in pieces

A level used to be one stream: every split bit, then three planes of
residual. That made a truncated level worthless — a prefix gave the red
channel and no green or blue — so the decoder threw it away and the
truncation curve was a staircase. Holding 80% of the file bought exactly
what holding 40% did.

A level is now a sequence of independent **units**, one per rectangle of
the level before, each carrying its own split bits followed by its own
rectangles' residuals interleaved across channels. A prefix is a whole
number of finished refinements; the units that never arrived keep the
rectangle and colour they had at the previous level. Units are emitted
largest-rectangle-first, an order both sides derive by sorting what they
already have, so nothing about it is transmitted.

    cut     before    after
    10%     19.55     21.79
    20%     19.55     23.07
    30%     19.55     24.45
    50%     25.15     25.62
    75%     25.15     26.01
    90%     25.15     26.13
    100%    26.22     26.22

Interleaving the residuals by rectangle costs nothing: measured at exactly
88,486 bytes either way. That is only true because the entropy layer moved
off deflate first — deflate leaned on runs of similar bytes, which the
planar layout provided and interleaving would have destroyed, while the
arithmetic coder's contexts are explicit and indifferent to grouping.

Largest-first ordering costs 160 bytes, 0.18%, from slightly worse
adaptation locality, and is worth up to +1.67 dB over tree order in the
first tenth of the stream. Against an oracle that orders units by actual
error reduction per byte — which cannot be shipped, since the decoder has
no way to know the error — area ordering captures 65% to 82% of the gain.

## The entropy coder

Levels are coded with an adaptive binary arithmetic coder (the LZMA range
coder) rather than deflate. Deflate models repetition; what the residual
planes hold is a distribution concentrated near zero, and deflate cannot
spend a fraction of a bit on a symbol it has no probability for.

The context model is derived on both sides from data already in the
stream, so nothing about it is transmitted — unlike spec §7, which budgets
2-6 MB for a shipped `context_model`. Three conditionings carry it:

- **Split flags** on the bucketed area of the rectangle they decide. A
  half-canvas block and a 2x2 tile have very different odds of splitting.
- **Residuals** on whether the parent rectangle subdivided. A rectangle
  that just split carries a genuinely new colour and a large delta; one
  that stayed carries a small correction to a colour already close.
- **Residuals** again on the magnitude of the co-located residual in the
  previous colour plane. A rectangle needing a large correction needs it
  in all three channels, and that is about *this* rectangle rather than
  about which regime it belongs to, so it survives the context above.

Ablated on `montanha_pessoas.jpg`, total container bytes:

    deflate                        116,260
    arithmetic, no contexts         99,637   -14.3%
    + residual split context        97,053    -2.6%
    + split-flag area context       93,970    -5.7%
    + cross-plane neighbour         87,060    -7.3%
    + k-means palette (below)       86,039    -1.2%
                                             -26.0% against deflate

Two predictions that led here were wrong, and the measurements are the
only reason that is known. The area context on split flags is worth more
than twice the residual context that motivated the entropy work. And the
neighbour conditioning was first tried on the *in-plane* predecessor,
since depth-first order keeps image neighbours close in the stream: that
was worth 0.04%, because whatever it predicts the split context already
said. The same idea across colour planes is worth two orders of magnitude
more.

Output is bit-identical across codecs, verified by decoding both
containers at all three levels and comparing every byte, in C and again
in JavaScript against the C output.

## Where detail goes

The subdivision gate used to measure deviation against a constant 255,
making it a test of absolute difference. The eye does not work that way:
an error of 7 on a pixel of value 16 is obvious, the same error on a pixel
of value 240 is invisible. The consequence was measurable and someone
spotted it by eye first — large dark regions collapsing into single
rectangles.

Leaf size by source luminance on `montanha_pessoas.jpg`, before:

    luminance    leaves   px per leaf   source's own deviation
      0-31         4563       9.5            16.3
     32-63         9322       6.5            33.8
     64-95        12954       6.3            37.8
    224-255         709      54.8             3.1

Shadows were getting leaves 50% coarser than mid-tones despite carrying
comparable detail. The deviation is now divided by
`255 * (luma + weber) / (pivot + weber)`, where `pivot` is the image's own
mean luminance — pivoting on a constant instead would tighten every dark
image and loosen every bright one, which is a quality setting wearing a
reallocation costume, and cost macarrao.jpg 1.35 dB when tried.

After, at `--weber 64`: shadows drop to 7.1 px per leaf, mid-tones stay at
6.1, and the sky coarsens from 54.8 to 79.1 — the bits move from a region
whose source deviation is 3.1 to one whose deviation is 16.3. Across
`samples/` it costs 0.4% to 6.9% more bytes at PSNR within 0.07 dB.

PSNR cannot see this improvement, by construction: it is an absolute-error
metric, and absolute error is exactly the thing the old gate was already
optimising. `--weber 1e9` restores the previous behaviour for comparison.

## Spending less on fine grain

Deviation says a region is not uniform. It does not say whether
subdividing would help. Distant grass varies as much inside a 4x4 window
as it does across the whole patch, so splitting reproduces noise nobody
could pick out; a face varies across the region and barely within a
window, so splitting is what resolves it. The ratio between the two — the
region's grain — comes from a 4x4 deviation map summed into an integral
image once per encode, so any region's is an O(1) query.

`--texture F` raises a region's minimum tile by `1 + F * grain`, which
spends grain at a coarser resolution and leaves structure at the full one.
At matched bytes on `montanha_pessoas.jpg` against plain tolerance
loosening, it moves 40% of the rectangles out of the highest-grain band
and into smooth and structured regions:

    band (source 8x8 deviation)   tolerance only      --texture 1
    smooth  <10                    95 @ 591.7 px     486 @ 109.4 px
    10-25                         561 @ 123.2 px    3473 @  19.6 px
    25-45                        2937 @  44.3 px    6898 @  17.0 px
    45-70                        7049 @  12.7 px    5588 @  17.1 px
    grain   >70                  6921 @   7.0 px    4157 @  14.1 px

**It is a rate control, not a free improvement.** Capping how fine
anything can get also caps quality: with it on, tightening the tolerance
saturates at 23.9 dB and cannot reach the 26.2 dB the default hits. The
crossover, measured:

    budget      tolerance only      --texture
    <= 60 KB   58.0 KB / 24.89 dB   36.7 KB / 23.87 dB    tolerance wins
    <= 40 KB   31.0 KB / 20.64 dB   36.7 KB / 23.87 dB    +3.2 dB
    <= 25 KB   14.8 KB / 19.33 dB   23.6 KB / 23.11 dB    +3.8 dB
    <= 15 KB   14.8 KB / 19.33 dB    7.0 KB / 19.61 dB    half the bytes

So it is off by default and is the right mechanism below roughly half the
default rate.

Two formulations failed before this one, both because they measured grain
by splitting a node and watching its children's deviation drop, which is
confounded with scale. Scaling the threshold by that drop only bit the
middle of the range and left the highest-grain band untouched at 6.1 px
per leaf. A hard floor on the drop pruned the root — the drop is small at
the top of the tree whatever the content — and collapsed the image to one
rectangle. The fixed 4x4 window is what removes the confound, and raising
the minimum tile rather than the threshold is what actually stops a
descent that a multiplier never could.

## The anchor palette

Candidates come from a 5-bit-per-channel histogram of the level-0 leaf
colours weighted by area, which bounds the work at 32768 bins whatever the
image size. A greedy max-salience pass picks the seeds — heaviest colour
first, then whichever candidate maximises weight x distance-to-nearest-
chosen — and twelve Lloyd iterations refine them.

The greedy pass maximises spread, which is good seeding but not the
objective; the objective is area-weighted assignment error, which is what
the refinement minimises. Worth +0.02 to +0.36 dB on the anchor across
`samples/`, and it shrinks the container too, since a better anchor leaves
smaller residuals behind it.

    $ head -c 5% image.nvdr > partial.nvdr
    $ nvdr_decode partial.nvdr out.ppm
    ... rendered at ANCHOR  (file carries only ANCHOR)  psnr 19.99 dB

The anchor is 1.5% of a typical container, so almost any surviving prefix
carries it.

## Build and run

    make
    ./nvdr_encode image.jpg out.nvdr
    ./nvdr_decode out.nvdr out.ppm --level 1 --compare image.jpg

`nvdr_encode` prints the raw and stored size and the PSNR of each layer,
because the point of this implementation is to test the spec's claims
rather than assume them.

A browser decoder lives in `public/nvdr.js`, with a viewer at
`public/nvdr.html` served by the repo's `server.js` — it shows the three
levels side by side and has a slider that truncates the container so the
guarantee can be watched rather than described.

## Measured

Container bytes at full quality, on `samples/`:

    image                   source   deflate     arith
    OIP-1304511485.jpg         24K     22.6K     18.5K   -18.1%
    OIP-3451121336.jpg         57K     71.2K     55.8K   -21.6%
    OIP-3786546191.jpg         48K     71.3K     54.8K   -23.1%
    OIP-4140498144.jpg         36K     56.7K     42.6K   -24.9%
    macarrao.jpg               13K     19.1K     15.7K   -17.7%
    montanha_pessoas.jpg      133K    112.8K     84.0K   -25.5%

Cumulative bytes and PSNR per level, montanha_pessoas.jpg:

    ANCHOR    3.1K  20.19 dB
    +R1      31.4K  25.11 dB
    +R2      84.0K  26.26 dB

Read this honestly: **JPEG wins on rate-distortion for photographs.** At
113 KB the source JPEG of montanha_pessoas sits far above 26.3 dB. What
this format buys is not a smaller file at equal quality — it is that the
first 3.7 KB are already a whole picture, and every byte after that
improves it without the decoder ever needing to wait or restart.

## Format

    [header 72B]
    [level 0 : palette, then split flags and anchor tokens, coded]
    [level 1 : units, each split flags + its rectangles' residuals, coded]
    [level 2 : units, each split flags + its rectangles' residuals, coded]

The palette rides ahead of level 0's coded stream: 48 bytes of genuinely
incompressible colour are not worth modelling. `--codec deflate` selects
the previous entropy layer, which the decoders still read.

Geometry travels as one bit per visited quadtree node — 1 splits, 0 stops —
and the decoder replays the subdivision rule from the canvas rectangle. No
coordinate is ever transmitted. Level k's bitstream is read per level-(k-1)
rectangle, so each level only describes where it disagrees with the last.

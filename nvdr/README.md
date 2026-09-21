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

## Colour

Residuals run in BT.601 YCbCr, with the two chroma channels quantised
`--chroma` times coarser than luma. The transform is fixed-point integer
on both sides, because encoder and decoder have to land on the same byte
and a float would be a portability bug waiting to happen. The
reconstruction is carried in YCbCr rather than re-derived from RGB at each
level: that round trip is lossy by a few units, and re-deriving would let
the drift accumulate.

Against RGB residuals on `samples/`, at the default tolerances:

    image                  source     RGB          YCbCr
    OIP-1304511485.jpg       24K   18.5K 22.62   13.3K 22.49   -28%
    OIP-3451121336.jpg       57K   55.5K 21.74   37.4K 21.60   -33%
    OIP-3786546191.jpg       48K   55.4K 24.04   36.3K 23.80   -34%
    OIP-4140498144.jpg       36K   45.1K 24.67   27.1K 24.43   -40%
    macarrao.jpg             13K   16.3K 29.14   10.0K 28.86   -39%
    montanha_pessoas.jpg    133K   86.6K 26.22   48.9K 25.90   -44%

Held to the same size instead of the same settings, YCbCr is +1.03 dB;
held to the same PSNR, it is 19% smaller. Note that most of this arrives
before any chroma coarsening: the plain transform at `--chroma 1` already
saves 30%, because R, G and B move together and coding them separately
pays for the same information three times. The cross-plane context was
worth 3.4% recovering exactly that redundancy after the fact — removing it
at the source is worth ten times more.

PSNR understates the case here. It weights the three channels equally
while the eye does not, so chroma coarsening costs more on the meter than
on screen. Measured by eye on the saturated regions, `--chroma 1` and `2`
are indistinguishable, `3` puts a faint magenta cast in the sky, and `6`
speckles colour across the backpack. The default is 2; `--chroma 0` drops
the transform and codes RGB, for comparison.

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

## Ramps

Every rectangle used to be a flat fill, and that is what set the format's
ceiling. On montanha_pessoas the flat encoder stops at roughly 26 dB and
stays there: tightening the tolerance to 0.050/0.020/0.004 and the step to
8/2 buys 79 KB of container and 26.25 dB, against 50 KB and 25.90 dB at the
defaults. More bytes do not help, because the thing being approximated is a
gradient and a flat rectangle cannot represent one at any size. The only
way out was to subdivide until each piece was small enough that its slope
did not matter, which is expensive in exactly the regions — sky, skin,
out-of-focus background — that ought to be cheap.

So a rectangle may now carry a one-axis colour ramp instead: a flag, an
axis bit, and one slope per channel, giving the total edge-to-edge change
in the chain's colour space. The DC stays the region mean the residual
already paid for, because the ramp has zero mean by construction, so the
two never compete for the same bits.

It is optional per rectangle, decided by rate-distortion. The encoder fits
both axes by least squares, re-evaluates each with the exact integer ramp
the renderer will use, and keeps the ramp only when the squared error it
removes beats `--gradient` times the bits it costs. A ramp that quantises
to all zeros costs four bits and buys nothing, so the comparison rejects it
on its own. **This version never loses to the flat one**: the flat fill is
always on the table and wins whenever it should.

Rate-matched against the flat encoder — tolerance tuned so the containers
come out the same size — across `samples/`:

    image                    flat              ramp
    OIP-1304511485.jpg   13640 B  22.49 dB  15711 B  23.57 dB
    OIP-3451121336.jpg   38310 B  21.60 dB  38605 B  22.33 dB
    OIP-3786546191.jpg   37209 B  23.80 dB  37162 B  23.87 dB
    OIP-4140498144.jpg   27748 B  24.43 dB  27691 B  25.44 dB
    macarrao.jpg         10215 B  28.86 dB  10343 B  29.43 dB
    montanha_pessoas.jpg 50030 B  25.90 dB  50397 B  26.80 dB

+0.07 to +1.08 dB at matched rate, and it arrives with a third fewer
rectangles, so there are fewer seams for the blend to clean up afterwards.

The slope step is deliberately coarse — 16, where the residuals run at 16
and 4. A slope is coded roughly unary, so its cost grows with its
magnitude, and a fine step prices the large ramps out of the
rate-distortion test. Those are precisely the ramps worth having. Swept at
matched rate on montanha_pessoas:

    step  lambda    bytes     PSNR
       1     150    56603    26.61 dB
       2     160    56051    26.59 dB
       6     200    56265    26.67 dB
      12     280    56577    26.92 dB
      16     320    55381    26.85 dB
      24     400    54460    26.77 dB

The lambda above is 600. The first cut of this shipped 280 together with
loosened tolerances, and that pairing was wrong in a way worth recording —
see **Why tolerance is not one knob** below.

Two alternatives lost. A full 2D plane fits better but needs two slopes per
channel and comes out behind per byte. Gating the tree on ramp fit rather
than mean fit — only splitting where a ramp would not do — makes the
surviving rectangles precisely the ones with large, expensive slopes.

Ramps live in the coded stream, so `--codec deflate` keeps every rectangle
flat and the two codecs are only comparable with `--gradient 0`.

## Why tolerance is not one knob

Ramps arrived with the default tolerances loosened from 0.090/0.040/0.018
to 0.140/0.070/0.040, on the reasoning that a rectangle able to follow a
gradient does not need to be subdivided as far. Measured across the six
photographs in `samples/` it looked like a straight gain. It was not.

Those three numbers are one step apart from each other, so the change did
not loosen the pyramid — it **shifted it down a level**. The new level 2
came out with exactly the rectangle count the old level 1 had, on every
image:

    image              tol .09/.04/.018      tol .14/.07/.04
    circulos         1018  1744  2308        196  1318  1744
    macarrao          670  4843  9475          1  2209  4843
    montanha_pessoas 9364 37663 53707       1930 17563 37663
    starfield           7 103891 606430        1    13 103891

A level of refinement was thrown away, and six photographs did not show it.

What hid it is that the tolerance knob does not mean the same thing on
different content. Rectangles produced as tolerance tightens:

    tolerance        0.200  0.140  0.090  0.060  0.040  0.018  0.010
    blocos              16     16     16     16     16     16     16
    circulos             1    196   1018   1474   1744   2308   2626
    macarrao.jpg         1      1    670   2746   4843   9475  11689
    montanha_pessoas     4   1930   9364  23167  37663  53707  57415
    starfield            1      1      7   4507 103891 606430 721771

A star field multiplies by 23 between 0.060 and 0.040 and has still not
saturated at 0.010, because noise has detail at every scale and there is no
tolerance at which it is resolved. There the knob is a smooth, powerful
rate dial: it lands anywhere on the curve, and PSNR degrades gracefully
because the error is spread — 80% of the squared error is spread over 22%
of the pixels.

A picture of flat shapes is the opposite. `blocos` is 16 rectangles at
every tolerance in the range: its structure is finite and the tree finds
all of it immediately. `circulos` moves by a factor of 1.5 across the whole
sweep. There the knob is a cliff, not a dial — below saturation it buys
nothing, and above it the edges go, which is where the picture is: 1% of
`circulos`' pixels carry half its squared error. Loosening cost it 5.5 dB
to save 7% of its bytes.

So tolerance is a rate control whose behaviour is a property of the image,
not of the format, and it cannot be retuned on photographs alone. It stays
where it was measured to belong, and the ramp is tuned separately through
`--gradient`.

The same asymmetry is why a star field looks like this codec's best case
and is really its worst. It compresses well and scores well, but its anchor
is **one rectangle** and its level 1 is thirteen — so the container has no
usable intermediate states at all. Truncating it gives 21.59 dB at 1% of
the file and 21.75 dB at 25%, flat across a quarter of the bytes, because
level 2 is delivered as one unit per level-1 rectangle and there are only
thirteen of them. The photograph next to it climbs 18.40 -> 19.85 -> 22.45
-> 25.46 dB over the same range. Good rate-distortion numbers on that image
are hiding the fact that the progressive ladder, which is the entire point
of the format, is not there.

## Softening the seams

A rectangle meets its neighbour at a hard step, and that step is the most
visible thing the format does wrong. Averaging every pixel with its four
neighbours at 0.40 each fixes the one-pixel band along every seam and
leaves everything else alone — inside a rectangle the neighbours carry the
same colour, so the average returns it unchanged. It is exactly a boundary
blend without needing to know where the boundaries are.

Costs no bytes and changes no format: `nvdr_smooth()` is a choice the
decoder makes, and `--smooth 0` renders the seams hard. Worth +0.00 to
+0.77 dB across `samples/`, never negative.

The obvious design was tried first and does not work. Scaling the blend
width by the colour difference across the seam — wide where the step is
big, nothing below a threshold — moved PSNR by 0.03 dB. A large colour
difference is usually a real edge, so widening the blend there smears what
should stay sharp, while the banding actually worth fixing sits in smooth
regions where the differences are small and the threshold skips it. The
rule is backwards. Scaling the width by rectangle size instead, on the
theory that a big flat block's seam reads as an artefact, measured 26.91
against the flat rule's 26.90 — no better either.

    no blend                             26.22 dB
    width by colour difference, max 8px  26.25 dB
    width by rectangle size, max 8px     26.91 dB
    one pixel, uniform                   26.90 dB

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

What the entropy coder is worth, at `--gradient 0` so both codecs carry
the same thing:

    image                   source   deflate     arith
    OIP-1304511485.jpg       23.7K     12.9K     10.6K   -17.8%
    OIP-3451121336.jpg       56.7K     44.5K     32.8K   -26.2%
    OIP-3786546191.jpg       48.4K     45.5K     33.2K   -27.1%
    OIP-4140498144.jpg       36.2K     31.0K     22.6K   -27.1%
    macarrao.jpg             12.7K      7.6K      6.0K   -20.9%
    montanha_pessoas.jpg    133.4K     52.0K     36.8K   -29.3%

At the defaults, against the same encoder with ramps off:

    image                     flat            ramps on
    OIP-1304511485.jpg   13640 B  22.49   18126 B  24.92 dB
    OIP-3451121336.jpg   38310 B  21.60   39852 B  21.83 dB
    OIP-3786546191.jpg   37209 B  23.80   38501 B  24.15 dB
    OIP-4140498144.jpg   27748 B  24.43   29253 B  25.20 dB
    macarrao.jpg         10215 B  28.86   10826 B  30.00 dB
    montanha_pessoas.jpg 50030 B  25.90   51765 B  26.30 dB
    blocos (synthetic)     156 B  52.64     160 B  52.64 dB
    circulos (synthetic)  3084 B  29.22    3851 B  30.19 dB
    linhas (synthetic)    4725 B  17.70   24334 B  36.75 dB

Cumulative bytes and PSNR per level, montanha_pessoas.jpg:

    ANCHOR    4.5K  19.55 dB
    +R1      17.1K  24.30 dB
    +R2      51.8K  26.30 dB

Read this honestly: **JPEG wins on rate-distortion for photographs.** At
133 KB the source JPEG of montanha_pessoas sits far above 26.3 dB. What
this format buys is not a smaller file at equal quality — it is that the
first 4.5 KB are already a whole picture, and every byte after that
improves it without the decoder ever needing to wait or restart.

## Format

    [header 72B]
    [level 0 : palette, then split flags and anchor tokens, coded]
    [level 1 : units, each split flags + its rectangles' residuals, coded]
    [level 2 : units, each split flags + its rectangles' residuals, coded]

A rectangle's residual is three signed deltas, followed by a ramp flag and,
when it is set, an axis bit and three slopes.

The palette rides ahead of level 0's coded stream: 48 bytes of genuinely
incompressible colour are not worth modelling. `--codec deflate` selects
the previous entropy layer, which the decoders still read.

Geometry travels as one bit per visited quadtree node — 1 splits, 0 stops —
and the decoder replays the subdivision rule from the canvas rectangle. No
coordinate is ever transmitted. Level k's bitstream is read per level-(k-1)
rectangle, so each level only describes where it disagrees with the last.

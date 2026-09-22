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

## Delivering a unit in pieces

A unit used to be all or nothing. Its split bits were emitted first and its
residuals after, so a decoder that ran out halfway had the shape and no
colours and had to throw the whole unit away. The truncation granularity of
a level was therefore one unit, and a level has exactly as many units as
the level before it has rectangles.

Interleaving the colours into the geometry removes that. Each rectangle's
residual and ramp are emitted the moment its split decision says it is a
leaf, so a cut anywhere leaves a valid partial subtree: what arrived keeps
its own colour and the region below the cut falls back to what its parent
was showing there. The symbols are the same ones in the same per-model
order, so the models adapt identically and the containers come out byte for
byte the same size — 4470 / 12538 / 34685 on montanha_pessoas either way.

Measured honestly, it is worth less than it looks. Across the sample set at
six cut points it moves nothing at all in most cells, and its two real
gains are at early cuts, where one unit is a large share of everything that
arrived:

    image                  12%    20%    30%    45%    65%    85%
    macarrao.jpg         +0.88  +0.07  +0.02  +0.01   0.00   0.00
    circulos (synthetic)     -  +0.89  +0.02  +0.39   0.00   0.00
    montanha_pessoas.jpg +0.21  +0.01   0.00   0.00   0.00   0.00
    everything else      +0.05  +0.01   0.00   0.00   0.00   0.00

This was built to fix the flat truncation ladder a star field showed —
21.59 dB at 1% of the container and 21.75 dB at 25% — and it does not fix
it. That image had thirteen rectangles at level 1 and therefore thirteen
units at level 2, and finer delivery inside those units is worth 0.16 dB
there. The flat ladder was the broken tolerance schedule described above,
not the delivery, and reverting the schedule is what fixed it. What
interleaving actually buys is that a cut can no longer discard work that
already arrived, which is worth keeping because it costs nothing.

## Measured but not built: cutting along a line

Everything here partitions into axis-aligned rectangles, so any boundary
that is not horizontal or vertical is approximated by a staircase, and the
staircase costs rectangles without bound as the tolerance tightens. On a
picture of four flat circles, 21% of the leaves sit on 0.76% of the area —
they are the outlines — and each of them has a deviation of 0.23 against a
tolerance of 0.04. They stopped splitting because they ran out of room, not
because they were resolved.

So the question is what one oblique cut would be worth against the 4-way
split the tree actually does. At every node that decides to split, with the
line searched over 8 angles and 7 offsets and both regions given their mean
colour:

    image                  erro removido      razao   custo   corte
                          corte     split                     vence
    circulos (synth)      37.5%     29.5%     1.27x   0.67x   88.8%
    blocos   (synth)      30.6%     53.3%     0.58x   0.67x    0.0%
    macarrao.jpg          36.6%     27.9%     1.31x   0.67x   76.0%
    OIP-4140498144.jpg    39.1%     31.6%     1.24x   0.67x   69.4%
    montanha_pessoas.jpg  33.9%     31.7%     1.07x   0.67x   63.2%
    starfield             34.1%     33.6%     1.01x   0.67x   74.5%

A cut removes 1.0 to 1.3 times the error of a full 4-way split while
costing roughly two thirds of the bits — two residuals and a line instead
of four residuals and four flags. It wins on 63 to 89% of decision nodes on
real images. `blocos` is the honest exception: its edges are axis-aligned
by construction, the quad split is exactly the right move there, and a
rate-distortion test would keep it.

Three caveats, because this is a ceiling and not a result. The line
parameters are unquantised and the region colours free, so real coding
costs more than the 10 bits assumed for angle and offset. The comparison is
one step deep: whether a cut actually prevents the staircase from forming,
rather than merely beating one split, cannot be measured without building
the alternative encoder. And the search is 56 passes over each node's
pixels.

A shape vocabulary — circle, ellipse, polygon — was the starting point for
this and is the narrower idea. On `circulos` it is worth far more: four
centres, radii and colours plus a background is about 31 bytes against the
3851 the container spends, a factor of 124. But it needs the shapes to be
found, which is a recognition problem rather than a coding one, and a
photograph contains none of them. The oblique cut is the same insight with
the recognition removed, since every boundary is locally a line, and it
composes with the tree, the levels, the truncation guarantee and the ramps
as they already stand.

## Measured: reusing the previous frame

The project's target is video, so the question that matters is what frame
N-1 is worth to frame N. There is no video here and nothing that can
decode one, so the frames are synthesised: a window cropped out of a
larger still and moved, which reproduces the camera translating, the
camera zooming, and a region moving against a still background.
`--noise` puts a deterministic per-pixel perturbation back to stand in for
a sensor. Read every number below knowing they are built rather than
filmed, so they are optimistic; the noise row is the honest one.

The first thing measured was the obvious primitive — copy the region when
it has not changed — and it does not work. Even between two **identical**
frames only 38% of the area can be copied and only 16% of the leaves
disappear, because the reference is the previous frame *decoded*, at
26.5 dB, and that loss already exceeds the tolerance across most of the
picture. Under a 3 px pan it collapses to 0.5%.

What does work is the thing the format already does between levels: code
the frame as a **residual against the previous frame's reconstruction**.
Measured end to end with the codec exactly as it stands — encode frame
N-1, decode it, subtract, encode the error, add it back:

    sequencia          intra              inter           bytes    PSNR
    static        40853 B  26.53 dB   12138 B  26.60 dB  -70.3%  +0.07
    object        40484 B  26.50 dB   13822 B  26.51 dB  -65.9%  +0.01
    pan           41643 B  26.52 dB   38668 B  26.43 dB   -7.1%  -0.09

Motion is not optional. A co-located reference is worth almost nothing the
moment the camera moves, and **one global motion vector** for the whole
frame fixes it:

    pan           41643 B  26.52 dB   16339 B  27.49 dB  -60.8%  +0.97
    pan + noise   42488 B  26.45 dB   17996 B  27.38 dB  -57.6%  +0.93

That is the crudest motion model there is — a single vector, found by
minimising absolute difference over a lattice. Per-block vectors can only
improve it.

The failure mode that decides whether this is a codec or a demo is drift:
frame 3 is predicted from a reconstruction of a reconstruction, and if
each step loses a little the picture walks away from the source. Over an
eight-frame chain, with the decoder's state carried forward exactly as a
decoder would hold it, it does not drift — it improves:

    quadro     bytes      PSNR    intra seria
        0      40853   26.53 dB      40853 B
        1      16339   27.49 dB      41643 B
        3      11086   28.04 dB      41578 B
        5       9958   28.29 dB      41299 B
        7       9078   28.41 dB      41224 B
                                     -63.2% total

Quality climbs 1.89 dB while the cost per frame falls to a fifth. That is
the truncation property working along the time axis rather than down the
container: every residual adds detail on top of what the last one left, so
a shot that holds still gets better and cheaper the longer it runs. With
sensor noise the same chain gives -57.9% and +1.79 dB, so the effect is
not an artefact of frames being perfect copies.

It is faster too, though less than it looks: the error image encodes in
25 ms against the frame's 46 ms, 1.8x, because at 26.5 dB the reference
still leaves plenty of structure behind.

Where this is still unproven: the frames are synthetic, the motion model
is one vector, the chain is eight frames rather than a shot, and nothing
here has been tried on footage with real lighting changes or a cut.

## Speed

Encoding runs at about 80 ms per megapixel, down from 136, with the
container byte for byte what it was — checked against a recorded hash on
every sample and on both video containers, because a speed change that
quietly moves the output is a quality change in disguise.

Where it went, measured on a 2.67 Mpx frame rather than guessed:

    fase                antes    depois
    arvore              163 ms   169 ms   (integral 17, desvio 92)
    rampa               157 ms   113 ms
    entropia             52 ms    53 ms

Two changes. The ramp fit swept its rectangle five times — once for the
flat error, once per axis for the least-squares fit, once per axis for the
exact error — and now sweeps it three, with the moments of both axes
accumulated together in the order they were summed before, since adding
the same doubles in a different order gives a different double. The fill
values of a candidate ramp depend only on the position along its axis, so
they are tabulated per rectangle instead of recomputed per pixel.

The tree used to compute every node's mean by adding up its pixels and
then sweep the region again for the deviation, which is two passes per
node at every depth. An integral image makes the mean four lookups. The
deviation still needs its sweep — a mean absolute deviation cannot be
recovered from sums — but it now accumulates the three channels as exact
integers and applies the luma weighting once at the end, instead of a
double multiply-add per pixel. That loop is the hot one in the whole
encoder, sweeping roughly ten times the image over a build.

Predicted right, measured wrong: the integral image was supposed to halve
the tree and moved it by 9%. The deviation sweep is what the tree actually
costs, and it is irreducible at 92 ms for 27 Mpx of reading.

The ramp fit is now threaded, which is the rest of it:

    imagem                  inicio    agora    ganho
    macarrao.jpg             82.7     44.1    46.7%
    montanha_pessoas.jpg    136.4     80.5    41.0%
    starfield               139.5     92.9    33.4%

Every rectangle's ramp is fitted from its own pixels into its own slot, so
the loop has no order and nothing shared. The container has to come out
identical anyway, and it does: checked against a recorded hash, and
checked for determinism over 70 runs across three images and a video
container, each producing exactly one hash. OpenMP is optional — without
it the pragma is ignored and the loop runs as it did.

ThreadSanitizer reports races here and they are libgomp's, not the code's.
Every report crosses the runtime boundary, where TSan has no visibility
into the barrier that `parallel for` ends with, and all of them disappear
at `OMP_NUM_THREADS=1` with the same binary. What clears it is not that
argument but the bit-identical runs; the argument only says why the tool
cannot see it.

The tree is not threaded. Its subtrees allocate nodes from one arena as
they descend, so running them in parallel makes the numbering depend on
which finished first — which is the bug the old pipeline had, and would
need per-subtree arenas merged in fixed order to avoid.

## Playing it

`public/nvdrv.js` is the sequence decoder in the browser, and the page has
a video tab that plays a `.nvdrv` at the source's frame rate. A video
dropped on the page goes to `POST /video`, where the server's ffmpeg cuts
it into frames, `nvdrv_encode` codes them, and the container comes back.

The JS decoder has to hold the same state as the C one after every frame,
not just draw something similar: a predicted frame is a residual on top of
that state, so one pixel apart at frame 1 is a different reference at
frame 2 and the difference compounds. `scripts/crosscheck_seq.mjs` compares
the two frame by frame, whole and cut at seven points, and the gate runs
it. Across 35 containers — every synthetic sequence, global and block
motion, three block sizes, short GOPs, ramps off, scene cuts — 280 checks,
no divergence. Moving the residual bias from 128 to 127 in the JS alone
fails the gate at frame 1.

### Against a real codec

Tested end to end with a real ffmpeg on a real VP8 file: a 960x540 clip
Playwright recorded from an animated page — a pan and a slow zoom over a
photograph, with a region moving against it — at 25 fps. Unlike the
synthetic sequences, it has the frame-to-frame noise a lossy codec leaves
behind. Checked first for the trap that would flatter the result: no
frame of the 144 is a duplicate, the smallest difference between
consecutive frames is 2.5 levels.

It is the first comparison against a video codec, and it loses:

    tolerancia            bytes    kbit/s     PSNR vs the VP8 frames
    0.090,0.040,0.018   2255789     3133      33.77 dB
    0.120,0.060,0.030    671234      932      30.47 dB
    VP8 itself           678729      943      (the reference)

At VP8's own bitrate, NVDRV reproduces VP8's frames at 30.5 dB. A
predicted frame costs 15 KB against an intra frame's 38 KB — 40%, where on
the synthetic sequences it was 25% and falling.

The obvious suspect was sub-pixel motion — the pan moves 1.12 px a frame
and the vectors are whole pixels — and it is not the cause. The best
quarter-pixel shift lowers the mean frame-to-frame error from 3.56 levels
to 3.24, 9%. What is left after the best possible alignment is about 3.4
levels everywhere: the previous codec's noise, which changes every frame.
A camera's sensor does the same. A quadtree of flat fills is the worst
possible representation of noise — following it means subdividing to the
pixel — and at the tolerances tuned for stills, that is what it does.

Past 0.12 the curve stops behaving: looser tolerances lose quality and
gain bytes (681 KB at 0.16, 685 KB at 0.22), which points at feedback in
the prediction loop — error left uncoded accumulating until it crosses the
tolerance — and is not yet understood. The default stays where it is
until it is.

### Real time

In the browser, on the same clip, a frame costs 21 ms to decode and 26 ms
to display against a 40 ms budget at 25 fps: not real time, and the
display is the larger half. The seam blend runs in JS over every pixel,
after a copy from RGB to RGBA, on the main thread. The decoder in C runs
at about 28 ms per megapixel, which is 38 fps at 720p and 17 at 1080p.

## Damaged input

Every test until this point fed the decoders valid files, or valid files
cut short. A file that crosses a network can also arrive damaged, and one
from an untrusted source can be damaged on purpose, so `make fuzz` builds
a harness that mutates real containers — bit flips, overwritten bytes,
header fields set to extremes, runs copied over other runs, truncation —
and decodes each result under AddressSanitizer and
UndefinedBehaviorSanitizer. Seeds cover every path the decoder has: arith
and deflate, with and without ramps, YCbCr and RGB, and a block-motion
sequence.

It found heap-buffer overflows in the still decoder on its first run, in
code that predates the sequence format. The per-rectangle arrays of a level
were sized from the leaf count in the header, while the rectangle list
itself grew on demand. A damaged split stream that subdivided past what the
header promised wrote colours straight past the end of those arrays, both
while replaying a unit and while filling in the units that never arrived.

Reading the decoder for the same pattern turned up five more of its
family that the fuzzer had not reached yet:

  - a split bit on a rectangle under two pixels wide is accepted, and the
    walk then descends through zero-area rectangles for as long as the
    bits say so
  - `anchor_bits` indexes a 256-entry model table and was never checked
  - the palette is copied by a count the stream supplies, without checking
    the stream holds that many bytes
  - the deflate layout's split and token lengths come from the header and
    were trusted against the inflated buffer
  - `ramp_offset` multiplies a slope step from the header by an axis length
    in `int`, which overflows on a damaged file — undefined behaviour, and
    a value the JS decoder would compute differently

All fixed on both sides, C and JS. A level that says something impossible
is dropped whole and the last good level is shown, since nothing after the
damage can be trusted to cover the canvas. The header is validated before
anything is allocated from it, with a hard ceiling of 134 Mpx. None of it
touches a valid file: every sample decodes to the same pixels, the
regression gate is unchanged on the stills, and C and JS still agree byte
for byte on every level and every cut.

The original crashing inputs are the proof. Fifteen of them — two still
containers and thirteen frame containers pulled out of sequences — overflow
the heap in the decoder before this change and decode cleanly after it.

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

## Sequences

`src/nvdrv.c` codes a sequence of frames, and the frame codec does not
change to make it work: a predicted frame is an ordinary NVDR container
whose image happens to be the prediction error, biased to the middle of
the range. Frame 0 is coded on its own; every frame after it is the error
against what the decoder holds, moved by one global translation.

Against coding every frame on its own, on the synthetic sequences:

    sequence      so intra              NVDRV                  bytes
    pan        984247 B  27.17 dB   245517 B  28.66 dB       -75.1%
    object     995208 B  27.08 dB   311216 B  27.07 dB       -68.7%
    pan+noise 1008011 B  27.09 dB   337152 B  28.58 dB       -66.6%

Truncation now holds on two axes. A prefix of the file is a prefix of the
movie, and the frame the cut lands in is handed to the still decoder as-is,
so it shows at whatever quality its bytes paid for:

    corte    bytes   quadros   PSNR medio
       5%    12275         1     24.75 dB
      15%    36827         1     27.12 dB
      30%    73655         4     28.04 dB
      50%   122758         9     28.40 dB
     100%   245517        24     28.66 dB

Scene cuts are found rather than declared: when the mean absolute
prediction error passes `--intra-thresh`, the frame is coded intra. On a
sequence spliced from two different sources the encoder puts an intra
frame exactly at the splice without being told where it is.

The encoder predicts from its own decoded output and never from the source
frame, because that is all a decoder has. Getting this wrong is the
classic way a codec drifts, and it is the reason the regression gate
checks a sequence: replacing the reconstruction loop with the source frame
makes the container **78.7% smaller**, which a size check would call an
improvement, and costs 8.90 dB by frame 12.

    ./nvdrv_encode frames/ out.nvdrv
    ./nvdrv_decode out.nvdrv --out played/ --compare frames/

A predicted frame's levels look broken and are not. Its anchor comes out
as one rectangle costing 9 bytes and its R1 as one more, with all 31585
rectangles in R2 — the same shape that made a star field's truncation
ladder flat. Here it is correct: a flat grey error means "no correction
yet", which renders as the previous frame, so a predicted frame cut short
falls back to what was already on screen. Tightening the tolerance does
restore a ladder, and costs 59% more bytes for 0.15 dB, so it stays as it
is. The anchor palette was the suspect before this was measured, and it
is not: 9 bytes is not where the waste would be.

### Per-block motion

One vector per frame describes a camera pan and nothing else. When
something in the shot moves differently from the camera, the global vector
is right for one of them and wrong for the other. Each 8x8 block now
carries its own vector, searched in two windows — around the global
vector, where most blocks are, and around zero, because what moves against
a pan is very often something holding still relative to the camera. The
field goes in the frame as deltas against the global vector, in two
deflated planes, so a shot with nothing moving against the camera pays a
few dozen bytes for it.

Measured with the field's cost inside the bytes:

    sequence      global              block 8
    pan        245517 B  28.66 dB   243331 B  28.76 dB     -0.9%
    object     311216 B  27.07 dB   289666 B  27.24 dB     -6.9%
    mixed      308443 B  28.16 dB   268886 B  28.53 dB    -12.8%
    pan+noise  337152 B  28.58 dB   333416 B  28.67 dB     -1.1%

Smaller and better on every sequence at once, for 18% more encode time.
Blocks of 16 and 32 were expected to win on vector cost and lost at every
row; a mostly-agreeing field deflates to almost nothing, so the finer grid
is nearly free. `mixed` — camera panning while a region moves the other
way — is the case one vector cannot describe, and it is only 4% of the
area there. Real footage, where far more of the frame moves independently,
is where this should matter most and is exactly what has not been measured.

Not here yet: B-frames.

## Layout

    src/        the codec: nvdr.c and entropy.c, plus nvdrv.c for sequences
    tools/      four CLIs: nvdr_encode/decode and nvdrv_encode/decode
    public/     the browser decoders (nvdr.js, nvdrv.js) and the page
    scripts/    the regression gate, the C-vs-JS cross-check, analysis
    samples/    the images every number in this file was measured on
    vendor/     stb_image.h, the only third-party code
    reference/  two modules kept out of a removed pipeline; not built

`GUIA.md` is the testing guide, in Portuguese, and is the shorter way in.

## Build and run

    make
    ./nvdr_encode image.jpg out.nvdr
    ./nvdr_decode out.nvdr out.ppm --level 1 --compare image.jpg
    ./nvdrv_encode frames/ out.nvdrv
    ./nvdrv_decode out.nvdrv --out played/ --compare frames/
    make check          # the regression gate

`nvdr_encode` prints the raw and stored size and the PSNR of each layer,
because the point of this implementation is to test the spec's claims
rather than assume them.

A browser decoder lives in `public/nvdr.js`, with a viewer at
`public/index.html` served by the repo's `server.js` — it shows the three
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

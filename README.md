# NVDR — Progressive Residual Stack

An implementation of the Progressive Residual Stack from NVDR spec
v0.14.1 §1.2, built for still images.

The spec describes a neural video codec: diffusion latents synthesised by a
TensorRT UNet, streamed over QUIC, read from NVMe through io_uring. None of
that is here, and none of it can be built or measured without a GPU and a
trained model. What *is* here is the part the spec is actually about — the
residual decomposition and the guarantee it provides — implemented on the
one domain that can be built and measured today.

## Format v10: the quadtree as the partition of a transform

Everything from "The decomposition" down to "The anchor palette" below
describes v9: flat rectangles in three levels. v9 was measured 6 to 11 dB
behind a DCT at the same size on every sample, because a flat rectangle
cannot hold texture (see "A texture layer, measured" and "The quadtree as
the transform's partition"). v10 keeps the quadtree and changes what a
leaf carries. The v9 codec is kept in `reference/nvdr_v9` so the numbers
measured on it stay reproducible. The decoders no longer read v9 files.

A v10 image is a quadtree of square leaves, 4 to 32 pixels, on 32x32
tiles. Every leaf carries:

- **a colour**, predicted as the rounded mean of the flat colours just
  above and to its left, plus a quantised correction. A leaf with nothing
  else is a v9 rectangle.
- **texture, if it pays**: the DCT of what the flat colour misses, at the
  leaf's own size, quantised with one step `q` and coded with the same
  adaptive arithmetic coder as everything else (a coded-block flag;
  significance and last flags by diagonal position; adaptive magnitudes).

The encoder picks the tree by rate and distortion. Each node is coded
whole and as four children against the live models, and the cheaper
in squared error + 0.12 q^2 * bits wins.

**Two layers, one guarantee.** Layer 0 is the tree and the colours, a
whole picture of flat blocks on its own. Layer 1 is the texture. Each is
one arithmetic stream coded tile by tile, and a cut stream decodes every
tile that arrived whole. A colour tile that did not arrive is grey. A
texture tile that did not arrive keeps its flat colours. Grey rather than
a predicted colour: on the `degrade` probe the predicted colour carried
the top rows down the image and a longer prefix scored worse, 12.5 dB to
9.1. Colour is predicted from flat colours only, never from texture, so
layer 0 never needs layer 1. Predicting from the full picture would save
0.3 to 2% of the bytes.

**Exact in C and in JS.** Everything the decoder computes is an integer:
HEVC's integer DCT (every matrix entry one of 33 values, shifts of 6 and
6 + log2 n, a 16-bit clip between the passes), 16-bit fixed-point
YCbCr to RGB, integer means. `scripts/crosscheck.mjs` finds the two
decoders identical at both layers and at cuts landing anywhere in either,
for q from 1 to 4000 and block limits down to 8..16.

At the default `q` 24, against v9 at its defaults (v9 PSNR as its
decoder showed it, seams blended):

    image                 v9                 v10
    OIP-1304511485     18126 B 25.51 dB    17345 B 34.95 dB
    OIP-3451121336     39852 B 21.85 dB    39947 B 31.47 dB
    OIP-3786546191     38501 B 24.38 dB    34025 B 32.16 dB
    OIP-4140498144     29253 B 25.71 dB    21696 B 33.62 dB
    macarrão           10826 B 30.71 dB     6222 B 39.55 dB
    montanha_pessoas   51765 B 27.00 dB    52809 B 33.28 dB
    degrade (probe)      598 B 33.83 dB      486 B 46.15 dB
    blocos (probe)       160 B 34.84 dB      315 B 56.34 dB
    circulos (probe)    2230 B 29.15 dB     3462 B 39.98 dB

The same size or smaller, and 6 to 10 dB better on every photograph.
Blocks and circles come out larger at the default only because the
default takes them to 40-56 dB. At equal quality v10 is smaller there
too: `blocos` 159 B at 38.4 dB (`--q 255`), `circulos` 1175 B at 29.7 dB
(`--q 96`). The codec lands 0.3 dB and about 3% short of the float
prototype in `scripts/analysis/qtdct`, the price of predicting from flat
colours and of the integer transform.

Two things got worse, and both are on the list:

- **the middle of the truncation curve.** Texture arrives tile by tile
  from the top, so half of montanha_pessoas shows its top half sharp and
  its bottom half flat, 24.0 dB against v9's 25.8. v9 spread its
  refinement over the whole picture. Sending the low frequencies of the
  whole image first would fix it.
- **decode speed in the browser.** A 960x540 video frame takes 63 ms to
  decode in JS, against 21 ms before. The display, which no longer blends
  seams, went from 26 ms to 3. The inverse DCT runs as plain matrix
  products; a butterfly and skipping all-zero rows are the obvious fixes.

Sequences use it as is. Every frame, intra or predicted, is a v10
container. Predicted frames set the header's residual flag, so their
colours are predicted as 128 rather than from neighbours. A residual's
neighbours say nothing about it, and predicting from them cost 13% more
colour bytes and 5% more texture bytes.

### Quarter-pixel motion (sequence format 5)

With whole-pixel vectors a predicted frame on the clean clip was 2.4 KB
of motion field and 9 KB of residual. VP8's whole predicted frame there
is 1.1 KB. The residual was the gap, and `scripts/analysis/subpel` found
why. A whole-pixel vector cannot follow a 1.12 px pan or a slow zoom, and
with a transform coding the residual, the misalignment it leaves is fine
texture across the whole frame, the most expensive thing there is to
code. Against the source frame, refining 16x16 vectors to half and then
quarter pixels took the residual from 4343 bytes to 1515 and PSNR from
40.0 to 43.2 dB.

Block vectors are now in quarter pixels, bilinear between whole pixels,
in integers the same way in C and JS: weights (4 - a) and a per axis,
rounded with + 8 >> 4, edges held. The search stays whole-pixel, then
refines each block to half and quarter pixels. The rate-aware pass that
weighs vectors against their bits runs on quarter-pixel SAD.

With quarter pixels the better block size became resolution-dependent.
On the 960x540 clip 16x16 wins: 591 bytes of field a frame against 2386,
for 0.16 dB. On the 128x128 gate sequence, whose moving object is 30 px,
8x8 wins both ways. So 16 from 0.2 Mpx up, 8 below. The vector's bit
weight rose from 8 to 16, since quarter-pixel deltas cost more bits.
Predicted frames default to a step 1.2 times the intra frames': across q
16 to 40 that was 0.1 dB better at equal rate.

The clean clip, 125 frames:

    NVDRV, v4 (whole pixels, q 24)             2400 kbit/s   34.35 dB
    NVDRV, format 5, q 20                      1013 kbit/s   37.80 dB
    NVDRV, format 5, q 24 (default)             822 kbit/s   36.86 dB
    NVDRV, format 5, q 32                       616 kbit/s   35.52 dB
    VP8                                         605 kbit/s   37.19 dB
    VP8                                         309 kbit/s   33.52 dB

At VP8's quality NVDRV needs about 1.5 times its rate, down from 11
times at the start of this work. The gate's 12-frame sequence is 15%
smaller at -0.42 dB, and at equal bytes it is 0.66 dB better. It is
encoded with equal quantisers now, so its drift check measures the
prediction loop and not the 1.2 ratio. The JS decoder takes 43 ms per
960x540 frame in Node, against a 40 ms budget at 25 fps.

### A sharper interpolation filter (sequence format 6)

Quarter pixels were first built with bilinear interpolation. Measured
between two source frames, H.264's 6-tap filter promised only 5 to 7%
less residual. Built into the loop it was worth far more, because
bilinear interpolation is a blur, and in a closed loop the blur
compounds. Each predicted frame is interpolated from a reference that
was itself interpolated from the one before. On the clean clip:

    interpolation        q     kbit/s    PSNR
    bilinear            24       690   36.22 dB
    bilinear            20       841   37.16 dB
    6-tap               24       532   36.03 dB
    6-tap               20       622   36.80 dB
    VP8                            605   37.19 dB

At equal rate the 6-tap filter is about 1 dB ahead of bilinear. It is
H.264's luma filter exactly, in integers, over the reference with its
edges held:

- half pixels across (B) and down (H): (1, -5, 20, 20, -5, 1) / 32,
  clipped;
- the centre (J): the same filter down a column of the unrounded
  horizontal sums, (sum + 512) >> 10;
- quarter pixels: the rounded mean of the two nearest whole or half
  samples.

The C side builds the three half-pixel planes once per reference, over
the frame plus a margin as wide as the field's largest vector.

The browser cannot afford whole planes per frame. `nvdrv.js` computes
per block only the samples that block's phase reads, over its own
region, gathered once into a contiguous buffer. The numbers are the
same, and the cross-check holds on every sequence. A 960x540 frame of
the panning clip, where every block has a fractional vector, decodes in
31 ms in Node, against 21 ms with bilinear and a 40 ms budget.

### Leaving still regions alone

A predicted frame used to correct every block: whatever small error the
reference carried, it was re-coded, and a little differently each
frame. On a background that does not move, that is visible as shimmer.
`scripts/analysis/flicker` measures it. Of the pixels whose source is the
same in two consecutive frames, it counts how many changed in the
decode. On a 40-frame clip with a still background and a moving subject,
5.5% of them changed every frame; with sensor noise added to the source,
8.7%.

Now, in a residual, the encoder also costs each leaf left uncorrected:
no colour correction and no texture, so the decoder shows exactly what
the previous frame held there. It picks that whenever it costs less in
error + skip_k * lambda * bits. The decoder needed no change, because an
uncorrected leaf was already a legal one.

    still background, moving subject     shimmer   bytes
    every leaf corrected                   5.47%   60622
    skip_k 0.25 (default)                  1.26%   48938   -19%
    same, sensor noise in the source       8.66% -> 2.05%   -26%

Leaving a block alone carries its reference's error forward, so the
weight matters on content where everything moves. On the panning clip,
0.25 sits 0.1 to 0.2 dB above the plain quantiser curve at equal rate.
1 falls 0.1 dB below it. The error a skipped block carries is bounded by
the same rule that skips it: once correcting pays, the block is
corrected. Per-frame PSNR falls about 0.2 dB over the first 25 frames of
a GOP and then holds flat.

The regression gate now encodes its sequence twice. The loop run uses
equal quantisers and no skipping, and holds drift to 0.25 dB, because
drift is how a reference that walks away from the source shows up. The
default run holds the encoder as it ships to the baseline, 25% smaller
than the loop run.

### Colour and seams in predicted frames

A predicted frame's residual was coded 4:4:4 and without the deblocking
filter, whatever the picture. Both were decided on the regression gate's
128x128 sequence, whose sawtooth texture is exactly the kind of picture
that wants neither. Now, when the intra frame chose 4:2:0 (every
photograph measured does), the residuals are coded 4:2:0 and filtered
too. The reference's colour is already smooth, so a residual coded whole
spends its bytes on colour detail and colour noise nobody sees. Over a
continuous prediction a residual's leaf seams show in the picture as
they are. The filter is the still decoder's, switched by the container's
own flag, so neither decoder changes. A sequence whose intra frame keeps
4:4:4 (a drawing, a screen, the gate) keeps both off.

BD-rate against the previous encoder, 48 frames, q 12 to 48
(`scripts/bench_video.mjs`'s NVDR half):

    clip                                        PSNR-Y   PSNR-RGB
    pan over a photo, whole pixels, clean        -8.2%     -5.1%
    subpixel pan, zoom, object crossing, clean   -6.2%     -0.3%
    the same with sensor noise every frame      -22.8%    -19.9%

The filter alone is 1.6% on luma and 2.4 to 3.1% on RGB. The rest is the
colour, which is worth most where the noise is: colour noise costs a
residual as much as detail does.

### Albums and the fluid context

The Fluid Codebook in `reference/codebook_db.c` persisted palette colours
across files. It was never measured cold against warm, and v10 has no
palette for it to feed. What an image codec actually learns from one
image and could reuse in the next is its adaptive models: how often a
leaf of each size carries texture, how its coefficients fall. Every
container starts those at even odds.

`NvdrContext` carries them. `nvdr_encode_mem_ctx` and
`nvdr_decode_mem_ctx` start from the models the last container left and
leave theirs behind. A context is only valid after a container decoded
whole, and an empty one behaves exactly like none. Measured cold against
warm:

    six samples in a row               -0.67%   (-1.5% on the best image)
    same at q 48                       -0.82%
    eight frames of the clean clip     -0.52%
    predicted frames of a sequence     ~0: +4% bytes and +0.2 dB, which
                                       sit on the same rate-quality curve

The gain is small because the models adapt within a few dozen symbols,
so learning from even odds costs an image of thousands of symbols little.
Sequences do not use it. The large win between images is in their
content, not in their statistics: a photo that repeats most of an
earlier one could be predicted from it the way a video frame is.

`.nvda` is the album: many images in one file (`src/nvda.h`, format 3).
Each image is coded one of two ways:

- **on its own**, as a v11 container;
- **predicted from an earlier image** of the same size, up to `window`
  places back (8 by default, at most 32). This uses the machinery of a
  sequence's predicted frames, exposed as `nvdrv_predict_encode` /
  `nvdrv_predict_decode`: block motion in quarter pixels with the 6-tap
  filter, the residual as a container with colours predicted as 128 and
  blocks that need nothing left alone.

The encoder ranks the earlier photos in the window by how much their
eighth-size grey thumbnails differ (with a shift of up to 2 thumbnail
pixels, for a pan), and codes a prediction from the best two. It
compares at equal quality: the prediction is coded with a finer step,
up to 2.4 times finer, until its squared error is within 0.1 dB of the
image coded alone, and is kept only if it is still smaller. At the same
step the prediction always won on bytes and lost about 1 dB, which is
not a comparison. On photos of the clean clip:

    album                                     alone      album
    six photos 0.4 s apart (pan + zoom)      168367 B   103338 B   -38.6%
    six photos 0.12 s apart (a burst)        169584 B    71702 B   -57.7%
    two scenes interleaved, window 8                              -65.7%
    the same, window 1 (only the one before)                       0%
    the six unrelated samples                                       0%

Referring to any earlier photo is what makes the interleaved album work:
a shoot that goes back and forth between two subjects never has the
similar photo right before. Unrelated photos are never predicted, so an
album is never bigger than its images coded alone.

#### One photo at a time, on a web page

An album on a site is not read front to back. A page shows one photo,
or a grid, and each photo has to come out as cheaply as a JPEG would.
So the file starts with an index, 16 bytes per image: where its payload
is, how long it is, its size, and which earlier image it was predicted
from. A reader that wants photo 10 reads the header and the index,
follows the references (10 from 8, 8 from 6, ...) down to a photo coded
alone, and fetches those payloads and nothing else, with HTTP range
requests. `nvda_decode` does the same in C with the bytes it has, holding
at most `window` decoded images.

The fluid context works against that: it makes every photo depend on
all the ones before it. It is worth 0.7% and is off unless `pack` is
given `--fluid`; a fluid album still decodes, only in order.

`public/nvdr-img.js` is the piece a site uses, an element in place of
`<img>`:

    <script type="module" src="nvdr-img.js"></script>
    <nvdr-img src="foto.nvdr" alt="..."></nvdr-img>
    <nvdr-img src="galeria.nvda#3" alt="..."></nvdr-img>

A single `.nvdr` is painted while it downloads, since every prefix of
the file is a picture: flat colour, then coarse texture over the whole
image, then detail. An album photo is fetched by ranges as above.
Elements showing photos of the same album share what they fetched, so a
grid of the whole demo album downloads 67.1 KB, the album's size, where
eight independent fetches of each chain came to 198 KB. A server that
ignores ranges sends the whole file, which still works. `server.js`
answers ranges; any static host (nginx, a CDN, S3) does too.
`public/galeria.html` is a page built that way, and `make demo` makes
its files.

#### Off the page's thread

A 13.5 Mpx photo takes 2.6 s to decode in JS, and on the main thread
that is 2.6 s of a page that neither scrolls nor answers a click. The
decoding now runs in module workers. `nvdr-tasks.js` holds the work
(decode a container or a prefix of it, open an album, decode one of its
photos, fetch one from a server by ranges); `nvdr-worker.js` runs it;
`nvdr-decoder.js` is the page's side, a promise per task and a few
workers, with a key always sent to the same one so what it keeps (a
container put once and cut at every slider position, an open album, the
ranges fetched from an album URL) is there next time. Pixels come back
transferred, not copied, and already averaged down to the size the
canvas is shown at, so the page does not shrink tens of megapixels on
its own thread either; the full picture comes too, for redrawing when
the page is resized. A browser that cannot start a module worker gets
the same results on the main thread: what was waiting, and what had
been put, is replayed there.

Measured as the longest stretch the page could not respond (Chromium's
long tasks), on the 13.5 Mpx photo:

                                   main thread   workers
    opening it                       3961 ms      154 ms
    moving the truncation slider     3007 ms        none

While a download streams, `<nvdr-img>` has the worker decode the longest
prefix that has arrived each time it is free, instead of queueing every
prefix: the demo's hero goes through 22 pictures on its way in, with no
long task. The slider likewise decodes only its latest position. The
video tab still decodes on the main thread: its frames take about 30 ms,
and the time it reports is meant to be the cost of playing the file
there.

`nvdr_album pack` and `unpack` make and open an album. `pack` prints, per
image, whether it was predicted and from which, its bytes, and what it
would cost alone; `unpack --only N` decodes one photo by its chain. On
the page, dropping several images at once sends them to `/album` and
shows the result; a `.nvda` file opens directly. The JS side
(`public/nvda.js`) mirrors both kinds, and `scripts/crosscheck_album.mjs`
holds it to the C pixels on every image, whole, cut, and one photo read
alone. The regression gate packs the photographs, where an album must
not be bigger, and the 12-frame sequence as a burst, where every image
after the first must be predicted. The fuzzer mutates an album's header
and index as well as its payloads, and reads its last photo alone.

### Frequency bands (format v11)

Texture used to travel as one layer, tile by tile from the top. Half of
montanha_pessoas therefore showed its top half sharp and its bottom half
flat: 24.1 dB, where v9, which spread its refinement over the whole
picture, gave 25.8. Texture now travels in two layers, the way
progressive JPEG's spectral selection does. The low frequencies of every
leaf come first, the rest after. A position in an n x n leaf is low when
u + v <= max(1, n * band / 32). `band` is in the header, 8 by default,
and 0 keeps all texture in one layer.

Swept over the six samples:

    band    bytes     PSNR, whole file   PSNR, first half of the file
      0    172044        34.40 dB           22.64 dB
      4    170849        34.32 dB           24.18 dB
      8    170330        34.28 dB           25.06 dB
     16    170525        34.27 dB           24.39 dB

montanha_pessoas cut at 30%, 50% and 75% goes from 22.2, 24.1 and
27.2 dB to 24.6, 26.5 and 28.8. The files come out about 1% smaller,
because the high band gets adaptive models of its own. The tiny
synthetic probes grow by the extra layer's fixed cost: `blocos` by 14
bytes.

Each band's inverse transform is rounded on its own. Clamping between
them lost 0.4 to 0.8 dB wherever the low band overshot 0 or 255 and the
high band could no longer pull it back. So the texture accumulates in a
16-bit plane and is clamped only when shown: `full = clamp(flat + low +
high)`.

Sequences keep `band` 0. Nobody watches a video frame arrive, and the
split costs bytes.

### Deblocking

Each leaf is quantised on its own, so where two leaves meet there is a
small step, and the eye finds a grid of small steps long before PSNR
does. With header flag `NVDR_FLAG_DEBLOCK` (on by default,
`--no-deblock` to leave it off), the decoder filters every leaf edge the
way H.264's normal filter does. A step smaller than alpha, with flat
enough pixels either side, is taken for quantisation and pulled together
by at most tc. Anything larger is a real edge and is left alone. alpha,
beta and tc are 20/16, 6/16 and 3/16 of the step. Swept on the six
samples, that gives +0.24 dB at q 24 and +0.28 at q 48; twice as strong
starts to cost.

Only edges with texture on at least one side are filtered. Between two
flat leaves the step is their two colours, and a colour's quantisation
error in pixels is step / n, a fraction of a level for any leaf larger
than 4. Filtering those edges anyway cost the `blocos` probe 6.7 dB, and
skipping them costs the photographs nothing.

In a sequence the filter runs on intra frames, whose decoded picture is
the next frame's reference. It does not run on predicted frames'
containers: those hold a residual, not a picture.

At the defaults, per sample: +0.14 to +0.47 dB at the same bytes.

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

### v11, and albums of large photos

A 3000x4500 photo took 13.6 s to encode and an album of three of them
56 s, 110 s when they repeated each other. Most of it was work done
again, or done where it could not pay:

    3000x4500, 4 cores                   before   after
    encode, with the report               13.6 s    6.0 s
    album, three unrelated photos         56.6 s   19.8 s
    album, a photo and two pans of it    110.6 s   47.4 s
    decode in JS, all three layers         7.8 s    2.6 s

Every container and album comes out byte for byte what it was, on every
sample, the burst and the demo album, and the gate's baseline did not
move.

- **The forward transforms leave the search.** A leaf's texture is the
  DCT of its pixels minus its flat colour, and a constant moves only
  the DC: every other row of the integer DCT sums to exactly zero (odd
  rows are antisymmetric, even rows fold into the half-size transform,
  down to the 4-point one). So a block's quantised texture, and the
  pixels each band of it adds, do not depend on what its neighbours
  predict. The encoder computes them for a whole row of tiles before
  searching it, on every core. The search itself stays one tile at a
  time, because each tile is decided against the models and colours the
  one before left.
- **-O3**, with the transforms' loops ordered so the innermost walks
  contiguous memory and each sum keeps its order: 25% on one core, the
  same doubles.
- **A residual leaf coded once, not twice.** Choosing between skipping a
  leaf and coding it coded it, and the winner was then coded again.
- **The report's three decodes run side by side.** `--quiet` skips
  them.
- **An album's motion search, once per candidate.** It does not depend
  on the step, and the equal-quality search tried up to six steps. A
  residual that already costs more than the photo alone is not decoded
  back.
- **Thumbnails too far apart are not tried.** A trial on a large photo
  is three times its encode. Predictions measured to win came up to a
  distance of 7.6 (a pan, 44 times smaller); pictures with nothing in
  common from 12.7. The cut is 10 (`--distance`).
- **The JS decoder** paints short rows by hand instead of a
  `TypedArray.fill` per row, copies a tile's state without making a
  subarray per row, and skips that copy when a layer arrived whole: a
  tile of it can then only stop on damage, and then the decode starts
  over the careful way. `scripts/crosscheck.mjs` now flips bits in
  full-length copies too, to hold that path to the C pixels. The page
  gets all three layers from one pass instead of three decodes.

## Against the browser's JPEG and WebP (v11)

`make bench` (`scripts/bench_codecs.mjs`) runs NVDR, and the JPEG and WebP
encoders a browser has (Chromium's `convertToBlob`), over their quality
ranges on the same pixels, scores every decoded picture here with one
implementation of each metric, and compares the curves by Bjontegaard
delta rate: how much bigger (+) or smaller (-) NVDR's file is than the
other's at the same quality, over the range both reach.

    image              NVDR vs JPEG                NVDR vs WebP              NVDR bits
                       PSNR-Y  SSIM-Y  PSNR-RGB    PSNR-Y  SSIM-Y  PSNR-RGB  Y / Cb / Cr
    OIP-1304511485     -15.5%   -6.1%  -49.8%      +57.1%  +80.6%   -2.8%    54 / 20 / 26
    OIP-3451121336     -25.4%  -21.8%  -34.6%       +3.7%   +5.7%  -11.5%    84 / 12 / 4
    OIP-3786546191     -19.4%   -9.2%  -34.7%       +8.2%  +17.1%  -13.2%    76 / 20 / 4
    OIP-4140498144     -29.4%  -19.8%  -33.8%       -0.6%   +6.7%   -9.3%    85 / 10 / 5
    macarrao           -36.7%  -31.3%  -46.5%       +2.7%   +4.8%  -11.8%    76 / 15 / 9
    montanha_pessoas   -29.5%  -21.3%  -33.4%       -3.2%   +0.9%   -8.8%    89 / 6 / 5
    montanha_ruido     -24.7%  -13.1%  -33.4%       -1.1%   +6.8%  -12.4%    87 / 7 / 6
    mean               -25.8%  -17.5%  -38.0%       +9.5%  +17.5%  -10.0%

NVDR beats the browser's JPEG everywhere, by a quarter on luma. Against
WebP it is even on luma for most pictures and ahead once colour counts,
and loses where colour is most of the picture. That is one choice, not
tuning: JPEG and WebP keep colour at half resolution each way (4:2:0),
which the eye barely sees and PSNR-Y does not see at all, and NVDR keeps
it whole, at 11 to 46% of its bits (last column, the encoder's own count
at q 24, now in its report). On OIP-1304511485, where the colour takes
46%, WebP needs 57% fewer bytes for the same luma.

Three things keep the comparison honest. Every codec gets the same
pixels, read once the way the encoder reads them and handed to the
browser as a PNG (the script checks the browser sees them). The samples
are JPEGs, and re-encoding a JPEG on its own 8x8 grid near its own
quality comes back almost exact: the browser's JPEG jumped from 35.6 to
49.6 dB between two steps. So JPEG sources are cropped 3 and 5 pixels
(`output/convert <in> <out> x0 y0`), off both grids. And montanha_ruido
is montanha_pessoas with high-ISO sensor noise added
(`scripts/analysis/addnoise.mjs`: spread growing with the light, mostly
luma, a little over a pixel wide), since a photo straight from a camera
is not a JPEG thumbnail. No AVIF: there is no encoder for it here.

### Colour at half resolution (4:2:0)

Header flag 0x04 gives Cb and Cr their own quadtree over a half-size
canvas: per 32x32 tile, luma's tree, then colour's 16x16 tree, in the
same three streams, so truncation and the progressive layers work as
before. The encoder averages colour 2x2 and weighs a colour sample's
error by four (it stands for four pixels); the decoders scale it back
with integer bilinear weights 9 3 3 1 in sixteenths, identical in C and
JS. The colour tree has its own split models.

`--chroma auto` (the default) picks per image. When halving the colour
costs little on its own (mean squared error under 2 per colour sample
after averaging and scaling back: every photograph measured), it is
4:2:0 straight away. Otherwise both modes are coded and the cheaper kept,
counting colour error at half luma's weight: at that weight every
photograph keeps 4:2:0 and every synthetic picture (blocos, circulos,
the gate's sequence) goes 4:4:4, where 4:2:0 cost blocos 21 dB. Album
photos predicted from another stay 4:4:4. A sequence's predicted frames
follow their intra frame (see "Colour and seams in predicted frames"):
halving the colour of residuals over a 4:4:4 reference drifted 1.3 dB
over the gate's twelve frames.

    BD-rate, mean of 7 images      vs JPEG            vs WebP
                                   PSNR-Y  SSIM-Y     PSNR-Y  SSIM-Y  PSNR-RGB
    4:4:4 (before)                 -25.8%  -17.5%     +9.5%  +17.5%  -10.0%
    4:2:0 forced                   -36.0%  -29.1%     -6.3%   -0.5%   -5.5%
    auto (default)                 -34.7%  -27.3%     -3.3%   +3.9%   -7.7%

Auto sits between the two because on the most colourful photo it keeps
4:4:4 at the high qualities, where its colour detail shows; `--chroma
420` forces it. At the default q the photographs come out 7-13%
smaller.

### Film grain synthesis

Sensor noise is the most expensive thing in a photograph to code and the
least worth keeping exactly: every grain is random, so the codec pays
full price for detail no one could tell from other grain of the same
size and strength. With `--grain auto` or `on` (`src/grain.c`) the
encoder measures the noise, filters it out, codes the clean picture, and
stores 22 bytes after the header (flag 0x08, their length in header
byte 29): the noise's strength at 16 brightnesses, how far a grain
spreads (one of five 3x3 kernels), and its strength in colour against
luma. Both decoders draw grain from AV1's 16-bit LFSR into a 64x64
template per component, read it at a hashed offset per 32x32 block and
lay it over the picture scaled by each pixel's brightness, in integers,
identically (crosschecked whole, cut and damaged; fuzzed).

Measuring noise on a photograph is the hard part, and what was learned:

- **Where.** In an 8x8 DCT, a block's level is the median of its 63 AC
  coefficients (texture is sparse, noise is not), and a brightness's the
  tenth percentile of its blocks (its flattest), each corrected by its
  bias for pure Gaussian noise (0.6745 and 0.8203, simulated). Reading
  the flattest blocks' pixels instead is right on flat noise and three
  times too strong on a textured photo; the DCT reading errs the other
  way on demosaiced noise (it has almost nothing at the top
  frequencies), so the strength used is the pixel reading capped at 1.8
  times the DCT's.
- **Shape.** Per-frequency noise measured directly let texture in; it is
  computed instead from the neighbour correlation, taking the noise as
  separable first-order autoregressive.
- **Physics.** Sensor noise is sigma^2 = a Y + b (photon noise plus read
  noise). Fitting that, weighed by block counts, keeps a night sky's
  stars (few, bright blocks) from reading as noise.
- **Filter.** Hard thresholding in the DCT (BM3D's first stage without
  its block matching) removed texture with the noise and came out 3.5 dB
  further from the clean picture than the noisy one; Wiener shrinkage,
  c^2 / (c^2 + n^2), came out closer than the noisy one.

On montanha at twice its size with sensor noise added (so the clean
picture is known), at the same q, against coding the noisy picture:

    q     without grain         with grain (distance to the clean picture,
                                           decoded before the grain)
    14    155031 B  33.84 dB    126025 B (-19%)  35.19 dB
    20    107368 B  33.89 dB     88825 B (-17%)  35.02 dB
    28     73732 B  33.75 dB     62406 B (-15%)  34.56 dB

Coding the noise keeps the picture 33.8 dB from the clean one however
many bytes are spent; filtering it gets closer with fewer. On a
3000x4500 night sky the file goes from 1.98 MB to 1.34 MB (-32%), and
the codec no longer turns the noise into blotches of blocks. The grain
laid back is not the source's grain pixel for pixel, and PSNR against
the noisy source says so; it looks the same.

Grain is off by default. `auto` turns it on when the measured level
passes 3.0: clean photographs at a camera's resolution read under 1.5,
the noisy test 3.9, the night sky 10; but small, densely textured
thumbnails are where texture and noise cannot be told apart (one clean
sample read 2.8, a crop of montanha 5.4). Residuals never carry grain
(a predicted frame would have to cancel its reference's), and albums and
sequences do not use it. The page has a selector for it.

### Measured but not built: directional prediction

H.264, HEVC and AV1 predict a block by extending its neighbours along a
direction, and it is worth 8-15% there. A prototype gave a leaf four
modes beside the average colour (vertical, horizontal, HEVC's planar,
the 135 degree diagonal), signalled per leaf. Luma PSNR as the encoder
reconstructs it, and bytes, at q 24:

                          no modes          from flat colours     from full reconstruction
                                            (best mode by RD)     (best mode by RD)
    montanha_pessoas      48427 B 33.89     48748 B 33.95         47951 B 33.99
    OIP-4140498144        19874 B 34.53     19984 B 34.58         19668 B 34.71
    montanha, 2x size     66047 B 38.34     66358 B 38.39         64827 B 38.38

Predicting from the neighbours' flat colours keeps layer 0 decodable on
its own and nets about 2%; choosing the mode by absolute difference
instead of by rate-distortion made files 1-3% bigger. Predicting from
the neighbours' full reconstruction, which would tie every layer to the
ones after it and so give up the truncation guarantee, nets 3-5%. The
quadtree already does most of what the modes do elsewhere: it splits to
4x4 exactly where an edge runs, and each leaf's transform is chosen by
rate-distortion, where HEVC's intra modes work on a fixed partition.
Neither gain pays for its format.

### Measured but not built: adaptive quantisation

x264 moves bits from busy regions, where the eye masks error, to smooth
ones, where it shows. A prototype gave each 32x32 tile a step offset of
-4..4 sixths of a doubling (H.264's QP scale), from how far its luma
activity (mean log variance of its 8x8 blocks) was from the picture's.
BD-rate against WebP, mean of the seven benchmark images, and on
montanha at twice its size:

                     7 images               montanha x2
    strength         PSNR-Y   SSIM-Y        PSNR-Y   SSIM-Y
    0 (off)          -3.3%    +3.9%         -31.5%   -31.2%
    0.5              -1.4%    +2.0%         -30.7%   -31.7%
    1                +2.1%    +3.6%         -29.6%   -32.1%

At a camera's scale it buys one point of SSIM for two of PSNR: a 32x32
tile of a large photograph mixes smooth and busy parts, and a per-leaf
offset would cost a symbol in every leaf. Not built.

### At a camera's scale

The benchmark's images are thumbnails of 400 to 800 pixels. On montanha
at twice its size, a picture closer to a camera's, NVDR against the
browser's WebP is -31.5% on PSNR-Y, -31.2% on SSIM-Y and -28.6% on
PSNR-RGB, where the thumbnails averaged -3.3%, +3.9% and -7.7%: the
quadtree's 32x32 leaves and the colour tree pay most where there are
large smooth areas to cover, and a photograph at full size has more of
them. `node scripts/bench_codecs.mjs <images>` runs any picture.

### The decoder as WebAssembly

`public/nvdr.wasm` (24 KB, `make wasm`) is the C decoder compiled for
the browser by clang and wasm-ld alone: no Emscripten, no wasi-libc.
`nvdr.c` leaves out its file and image-format code under `NVDR_WASM`;
`wasm/libc.c` supplies the memory functions (a stack the page rewinds
between decodes) and `wasm/api.c` the three calls the page makes; the
linker drops the encoder, and the module imports nothing.
`public/nvdr-wasm.js` loads it and installs it behind `decode()` in
nvdr.js, so everything that decodes (the page, the workers, albums,
`<nvdr-img>`) uses it; a decode with a fluid context, or any decode
where WebAssembly cannot load, stays in JavaScript.

    decode, 3000x4500           JavaScript   WebAssembly
    full picture                  2108 ms       856 ms
    with grain                    2840 ms      1470 ms
    three views (the page)        3690 ms      1919 ms
    montanha, 768x512              145 ms        26 ms

It is the same C the command line runs, so the three decoders are held
to one another: `scripts/crosscheck.mjs` now compares C, JavaScript and
WebAssembly on every layer, cut and damaged copy. One change went into
the C decoder with it: like the JavaScript one, it no longer saves each
tile of a texture layer that arrived whole (only damage can stop one
mid-tile, and then the decode starts over saving them), which was a
tenth of its time.

## Showing a large picture small

A night sky shown on the page at a tenth of its size came out as white
noise, and a grid of album photos as worse. The pixels were right: C and
JS agreed on them. The canvas was not. Every canvas carried
`image-rendering: pixelated`, which is right for enlarging and, for
shrinking, keeps one pixel of every 10x10 and drops the rest. On a
smooth picture that passes; on grain it keeps a scatter of single
grains, the brightest showing as white dots. Only the full layer has
the grain, so only the third view showed it, and a thumbnail, shrunk
further, more.

`paintFitted` in `nvdr.js` now draws a picture at the size the canvas is
shown, in device pixels, averaging every source pixel an output pixel
covers, and draws it again when that size changes. The page, the album
grid and `<nvdr-img>` all use it.

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

Two early explanations are wrong, and the measurements that ruled them out
are worth keeping. Sub-pixel motion is not the cause: the best
quarter-pixel shift lowers the mean frame-to-frame error from 3.56 levels
to 3.24, 9%. Noise is not the cause either. A clean clip, rendered
deterministically from the same page (`window.render(t)`, 125 frames,
JPEG q100, no codec in between), loses worse. Both codecs are measured
against that source with `scripts/analysis/psnr_dirs`:

    VP8 target   kbit/s   PSNR        NVDRV tolerance      kbit/s   PSNR
    150k            174   30.57 dB    0.200,0.100,0.050       925   28.72 dB
    300k            309   33.52 dB    0.120,0.060,0.030       947   29.85 dB
    600k            605   37.19 dB    0.090,0.040,0.018      3402   33.36 dB
    1200k          1202   39.41 dB    0.050,0.020,0.008      6174   34.45 dB

At equal quality NVDRV spends about 11 times the bits, and it has a floor:
no tolerance takes it under about 925 kbit/s.

`scripts/analysis/webm_frames.py` reads VP8's frame sizes out of the WebM,
and they put the gap in one place:

                       intra frame    predicted frame
    VP8 300k             55333 B          1099 B
    NVDRV default        56993 B         16029 B

The intra frames are the same size. They are not the same quality (see
"The intra frame is the gap", below). The predicted frames, which are
124 of 125, are fifteen times larger. `scripts/analysis/loopcost` explains why: at
the default tolerance, a predicted frame coded against the *source*
previous frame costs 91 bytes, and coded against the *reconstructed*
previous frame, as the closed loop must, about 20 KB. The genuine change
between frames is 3.6 levels. The reference's own coding error is 5.9.
Between 97% and 100% of every predicted frame is the codec re-coding
texture the intra frame left out. Flat rectangles cannot converge on
texture, so the error never goes away, and every frame pays for it again.
That is the rate floor. "Re-coding" is not the same as "wasted", though:
the same section below shows those bytes are refining an intra frame
that starts 7 dB behind VP8's.

Loosening only the predicted frames (`--pred-tolerance`, intra left at
the default) stops that re-coding. Every loose setting gives the same
file, 999 kbit/s at 30.29 dB. A predicted frame is then 91 bytes of
residual and 3607 bytes of motion field. The field is now the cost, and
it is coded naively, as deltas from the global vector:

    block    kbit/s   PSNR
    8           999   30.29 dB
    16          473   28.20 dB
    32          345   26.00 dB
    global      405   20.19 dB

So the gap has three parts, in order of size. The residual is decided by
a tolerance, not by rate against distortion, so predicted frames
re-code errors they cannot remove. The motion field ignores that
neighbouring vectors agree. And there is no rate control. What to fix
follows from that: predict each vector from its neighbours (done in v3,
under "Coding the field"), decide the residual per block by rate and
distortion (tried, and it measured something more basic, under "The
intra frame is the gap"), and add a zoom/affine global model. `--pred-tolerance`
stays off by default, because on its own it trades 3 dB for the rate.

### The intra frame is the gap

The next planned fix was to decide each block's residual in a predicted
frame by rate against distortion, and drop what does not pay. The
prototype coded the frame, decoded it, and measured what the residual
bought in each 16x16 block. It set the losing blocks to the neutral 128
and coded the frame again. It did not survive measurement, for two
reasons that are worth more than the fix would have been.

The residual pays for itself. Over the first predicted frames of the
clean clip it removes 140 units of squared error per bit on average,
and 55% of blocks gain. Per-frame PSNR shows what it is doing: the intra
frame comes out at 31.2 dB, and the predicted frames lift the clip to
33.4 dB within a few frames and hold it there. Dropping the residual
holds the clip at the intra frame's quality instead, 30.7 dB.

Dropping part of it drops all of it. The tolerance is a mean deviation
over a region, and the texture error in the residual, about 6 levels,
sits just above it, about 4.6. With half the blocks set to 128, every
region that straddles both falls under the tolerance and stops
splitting, the blocks that were meant to keep their residual included.
Every lambda from 0.01 to 300 gives the same collapsed frame: 300 bytes
of residual, 30.7 dB.

The comparison that matters is the intra frame against VP8's key frame,
measured on frame 0 of the same clip:

                                  bytes    PSNR
    NVDR, default tolerance       57113   29.6 dB (flat render; 31.2 in the sequence)
    NVDR, 0.03/0.012/0.004        86975   30.1 dB
    NVDR, same, --min-tile 1     239924   34.1 dB
    VP8 300k key frame            55333   38.3 dB
    JPEG, Chromium, quality 0.5   48117   37.2 dB

The JPEG number is flattered, since the source is itself a JPEG on the
same 8x8 grid. The VP8 one is not. Tightening the tolerance buys almost
nothing, because 2x2 flat rectangles cannot represent texture finer than
2 pixels. Allowing 1x1 buys 4 dB for four times the bytes. A transform
coder at the same size is 7 to 8 dB ahead. That is a property of
piecewise-constant approximation, not of any tuning. Every predicted
frame inherits it: its residual is the same texture error, coded by the
same rectangles.

### Where the finest level's bytes go

R2 costs 2.1 to 3.8 times R1's bytes. On every sample it is about 70%
of the file, for +1.3 to +4.3 dB. The rectangle count grows much less,
1.1 to 2.2 times, so most of the cost is not the new tiles. Counting
bits by category in the coder:

- every rectangle R1 hands down that does not split still pays a colour
  correction at R2's finer step, and 91-97% of them need one. That is
  40-80% of R2's bytes. The corrections are not waste: dropping them
  costs 1.1 to 2.2 dB. Making R1's step finer only moves the same cost
  into R1, and 16,4 stays the smallest file on every sample.
- the new tiles R2 subdivides into are what barely pay: 4-7 KB for
  +0.16 to +0.5 dB on most samples. That is the quadtree trying to hold
  texture with flat fills.

### A texture layer, measured

`scripts/analysis/texture` renders the rectangles at some cut and codes
what is left with an 8x8 DCT through the same arithmetic coder. PSNR at
the byte size of today's container, interpolated along each curve:

    image                  NVDR today       DCT alone        anchor + DCT     R1 + DCT
    OIP-1304511485         18126 B 24.92   33.28 (+8.36)    29.91 (+4.99)    29.21 (+4.29)
    OIP-3451121336         39852 B 21.83   30.06 (+8.23)    28.22 (+6.39)    26.18 (+4.35)
    OIP-3786546191         38501 B 24.15   32.15 (+8.00)    30.26 (+6.11)    28.66 (+4.51)
    OIP-4140498144         29253 B 25.20   34.70 (+9.50)    32.43 (+7.23)    30.86 (+5.66)
    macarrão               10826 B 30.00   41.02 (+11.0)    38.35 (+8.35)    35.74 (+5.74)
    montanha_pessoas       51765 B 26.30   31.93 (+5.63)    30.93 (+4.63)    29.68 (+3.38)
    f0000 (video bench)    57113 B 29.60   38.56 (+8.96)    34.92 (+5.32)    32.98 (+3.38)

The DCT alone, which is essentially JPEG with a better entropy coder,
lands where Chromium's JPEG does on f0000, so the prototype is
calibrated. The result is uncomfortable. Every rectangle added under
the DCT makes it worse. The hard edges of a flat-filled base end up in
the residual, and a fixed 8x8 transform pays dearly for edges that cross
its blocks. Laid on a grid, the rectangles are a cost to the transform,
not a help. Where the quadtree can still earn its place is as the
partition the transform runs on: large blocks where the image is smooth,
small ones where it is busy, each leaf carrying its colour plus the
coefficients of its own texture. A flat fill is then just a leaf with
no AC coefficients. `comparacao_macarrao.png` shows the images side by
side.

### The quadtree as the transform's partition

`scripts/analysis/qtdct` gives the quadtree a new job. Its leaves are
squares of 4 to 32 pixels on a power-of-two grid. Each leaf's colour is
predicted from the decoded pixels just above and to its left, so it
costs nothing to send, and what the prediction misses is coded with a
DCT of the leaf's own size. A leaf with no AC coefficients is exactly a
flat rectangle; the old codec is the special case where texture is never
sent. Each node splits only if its four children, coded against the
adaptive models as they stand, beat it on distortion + lambda * bits.

PSNR at today's container size, same deadzone (0.1) for both DCTs:

    image                  NVDR today       DCT 8x8 fixed   quadtree DCT            NVDR's PSNR in
    OIP-3451121336         39852 B 21.83   31.13            31.32 (+0.19)            23% of the bytes
    macarrão               10826 B 30.00   41.97            45.69 (+3.72)            13%
    montanha_pessoas       51765 B 26.30   32.42            33.20 (+0.79)            25%
    f0000 (video bench)    57113 B 29.60   39.11            41.24 (+2.13)            15%

(deadzone 0.33, all seven samples: +0.2 to +3.7 dB over the fixed grid,
NVDR's quality in 14-26% of the bytes.)

The quadtree beats the fixed grid on every sample, most where the image
has large smooth areas and least on dense texture. Two controls:

- without the prediction from the neighbours (`--pred none`) the gain
  over the fixed grid drops by 0.3 to 2 dB, and goes negative on the
  most textured sample. Predicting the leaf's colour is what makes big
  leaves cheap.
- with the AC coefficients switched off (`--dc-only`), which is flat
  rectangles again at a 4 px minimum, PSNR stops at 27.5 dB on macarrão
  however many bytes it is given.

lambda barely matters (0.06 to 0.3 times Q^2 moves nothing past 0.05 dB).
A 32 px maximum beats 16 by up to 0.5 dB. Chroma at the luma step
(`--chroma-q 1`) beats 1.5 slightly. `quadtree_dct.png` shows both images
at equal size. At about a sixth of the bytes, the quadtree DCT matches
today's PSNR. The artefacts it shows there are blocking in the largest
leaves, which a deblocking filter is for, not the staircase of flat
tiles.

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
    ./nvdr_album pack album.nvda a.jpg b.jpg c.jpg
    ./nvdr_album unpack album.nvda out/ --compare .
    ./nvdr_album unpack album.nvda out/ --only 3

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
field went in the frame as deltas against the global vector, in two
deflated planes, so a shot with nothing moving against the camera paid a
few dozen bytes for it. Version 3 codes it predictively instead; see
below.

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

### Coding the field

On the clean benchmark clip, once predicted frames stopped re-coding the
intra frame's texture, the motion field was most of what was left: 3607
of a predicted frame's 3714 bytes. Deflate over deltas against the global
vector sees runs of equal bytes. It cannot see that the likeliest vector
is the one next door.

Version 3 predicts each vector from the median of its left, top and
top-right neighbours, as H.264 does, and codes it with the same adaptive
arithmetic coder as the stills. A block that matches its prediction costs
one bit, conditioned on whether its left and top neighbours matched. Any
other block codes the difference, x then y, as zero flag, sign and
adaptive unary magnitude.

A better coder for the field does little while the search ignores it,
because the search picks each vector by prediction error alone. Among
near-equal matches on texture, the winner flickers from block to block.
A second pass therefore walks the blocks in coding order, when each
block's prediction is known. It weighs a handful of candidates (the
search's pick, the prediction and its four neighbours at one pixel, the
global vector, the left and top vectors) by block error plus
`--mv-lambda` times the bits the vector would cost. That pass is
sequential, because every choice moves the prediction of the blocks
after it. It is cheap, because it only evaluates candidates.

On the clean clip, against v2:

                               v2                    v3, lambda 8
    default tolerance     3402 kbit/s 33.36 dB    3143 kbit/s 33.39 dB   -7.6%
    --pred-tolerance loose 999 kbit/s 30.29 dB     816 kbit/s 30.36 dB  -18.3%

    field per predicted frame, loose:  3607 B  ->  2674 B

The coder alone, with lambda 0, gives 905 kbit/s. The rest comes from the
rate-aware pass. Larger lambdas keep trading quality for rate (715 kbit/s
at 16 and 30.17 dB; 593 at 32 and 29.81 dB). 8 is the default because it
is the largest that costs no quality at the default tolerance. The
12-frame sequence in the regression gate is 4.1% smaller at -0.02 dB.

That is not the gap to VP8 closed. The field is still about 2.3 bits per
8x8 block. VP8 spends its whole predicted frame, 637 bytes at 150 kbit/s,
on 16x16 macroblocks that split only where it pays. The next step for
the field is the same thing in this codec's own terms: a quadtree over
the motion that stays whole where the vectors agree. At the default
tolerance the residual is still 12 KB of the 15 KB frame, and that is
fix (b).

Not here yet: B-frames.

## Layout

    src/        the codec: nvdr.c (v10) and entropy.c, plus nvdrv.c for sequences
    tools/      five CLIs: nvdr_encode/decode, nvdrv_encode/decode, nvdr_album
    public/     the browser decoders (nvdr.js, nvdrv.js) and the page
    scripts/    the regression gate, the C-vs-JS cross-checks, the fuzzer, analysis
    samples/    the images every number in this file was measured on
    vendor/     stb_image.h, the only third-party code
    reference/  the v9 rectangle codec and two modules from a removed
                pipeline; not built

`GUIA.md` is the testing guide, in Portuguese, and is the shorter way in.

## Build and run

    make
    ./nvdr_encode image.jpg out.nvdr [--q 24]
    ./nvdr_decode out.nvdr out.png --layer 1 --compare image.jpg
    ./nvdrv_encode frames/ out.nvdrv
    ./nvdrv_decode out.nvdrv --out played/ --compare frames/
    ./nvdr_album pack album.nvda a.jpg b.jpg c.jpg
    ./nvdr_album unpack album.nvda out/ --compare .
    make check          # the regression gate
    make fuzz           # the decoders under ASan and UBSan

`nvdr_encode` prints each layer's stored size and the PSNR of the file
read back at that layer, and how many leaves of each size the tree chose.

`public/index.html`, served by `server.js`, shows the three layers side by
side, a slider that truncates the container, and plays sequences.

## Format (v11)

    [header 32B]  "NVDR", 11, flags, width u16, height u16, max block u8,
                  min block u8, luma step u16, chroma step u16,
                  layer 0/1/2 bytes u32 x3, band u8, 3 reserved
    [layer 0]     per 32x32 tile in raster order: split flags, and per
                  leaf three colour corrections
    [layer 1]     per tile, per leaf in the same order: three low-band textures
    [layer 2]     the same for the high band; empty when band is 0

Flags: 0x01 residual (every colour predicted as 128), 0x02 deblock.

The canvas is padded to a multiple of the smallest block. A node that
runs past it has no split flag and always splits; one wholly past it does
not exist. The v9 format this replaced is described in the git history of
this file and implemented in `reference/nvdr_v9`.

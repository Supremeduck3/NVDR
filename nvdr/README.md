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
    [level 1 : split flags and 3 int8 residual planes, coded]
    [level 2 : split flags and 3 int8 residual planes, coded]

The palette rides ahead of level 0's coded stream: 48 bytes of genuinely
incompressible colour are not worth modelling. `--codec deflate` selects
the previous entropy layer, which the decoders still read.

Geometry travels as one bit per visited quadtree node — 1 splits, 0 stops —
and the decoder replays the subdivision rule from the canvas rectangle. No
coordinate is ever transmitted. Level k's bitstream is read per level-(k-1)
rectangle, so each level only describes where it disagrees with the last.

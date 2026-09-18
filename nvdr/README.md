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

The context model is derived from the geometry on both sides, so nothing
about it is transmitted — unlike spec §7, which budgets 2-6 MB for a
shipped `context_model`. Two contexts carry it:

- **Split flags** are conditioned on the bucketed area of the rectangle
  they decide. A half-canvas block and a 2x2 tile have very different odds
  of subdividing.
- **Residuals** are conditioned on whether the parent rectangle subdivided.
  A rectangle that just split carries a genuinely new colour and a large
  delta; one that stayed carries a small correction to a colour already
  close. Deflate coded both with the same model.

Ablated on `montanha_pessoas.jpg`, total container bytes:

    deflate                        116,260
    arithmetic, no contexts         99,637   -14.3%
    + residual split context        97,053    -2.6%
    + split-flag area context       93,970    -5.7%
    + both (shipped)                91,386   -21.4% against deflate

The coder itself is most of the win. Worth noting against the prediction
that led here: the area context on split flags turned out to be worth more
than twice the residual context that motivated the work.

Output is bit-identical across codecs, verified by decoding both containers
at all three levels and comparing every byte.

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
    OIP-1304511485.jpg         24K     22.7K     18.8K   -17.4%
    OIP-3451121336.jpg         57K     71.6K     57.3K   -20.0%
    OIP-3786546191.jpg         48K     72.2K     57.1K   -20.9%
    OIP-4140498144.jpg         36K     56.7K     44.7K   -21.2%
    macarrao.jpg               13K     19.2K     16.8K   -12.2%
    montanha_pessoas.jpg      133K    113.5K     89.2K   -21.4%

Cumulative bytes and PSNR per level, arithmetic, montanha_pessoas.jpg:

    ANCHOR    3.1K  19.99 dB
    +R1      35.6K  25.10 dB
    +R2      89.2K  26.26 dB

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

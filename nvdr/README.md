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

Each level is deflated on its own rather than the container being
compressed as a whole. That is what lets both properties hold at once: a
prefix of a single deflate stream does not decode, so one stream across all
three levels would buy a smaller file by destroying the only thing this
format is for. Per level, the bytes on disk are the bytes on the wire and
every prefix still ends on a boundary that decodes.

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

Cumulative container bytes and PSNR against the source, on `samples/`:

    image                   source     ANCHOR            +R1             +R2
    OIP-1304511485.jpg         24K   2.3K 18.0dB   10.3K 21.8dB   22.7K 22.6dB
    OIP-3451121336.jpg         57K   5.4K 18.9dB   37.1K 21.5dB   71.6K 21.7dB
    OIP-3786546191.jpg         48K   5.0K 20.1dB   34.1K 23.4dB   72.2K 24.1dB
    OIP-4140498144.jpg         36K   3.3K 19.6dB   23.9K 24.0dB   56.7K 24.7dB
    macarrao.jpg               13K   0.4K 18.4dB    5.8K 27.0dB   19.2K 29.2dB
    montanha_pessoas.jpg      133K   3.7K 20.0dB   44.7K 25.1dB  113.5K 26.3dB

Read this honestly: **JPEG wins on rate-distortion for photographs.** At
113 KB the source JPEG of montanha_pessoas sits far above 26.3 dB. What
this format buys is not a smaller file at equal quality — it is that the
first 3.7 KB are already a whole picture, and every byte after that
improves it without the decoder ever needing to wait or restart.

## Format

    [header 72B]
    [level 0 : palette, split bitstream, packed anchor tokens]   deflated
    [level 1 : split bitstream, 3 int8 residual planes]          deflated
    [level 2 : split bitstream, 3 int8 residual planes]          deflated

Geometry travels as one bit per visited quadtree node — 1 splits, 0 stops —
and the decoder replays the subdivision rule from the canvas rectangle. No
coordinate is ever transmitted. Level k's bitstream is read per level-(k-1)
rectangle, so each level only describes where it disagrees with the last.

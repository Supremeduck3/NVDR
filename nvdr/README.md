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

In the spec, `L` is a 64x64x4 diffusion latent and the three layers are
int4/int8/fp16 quantisations of it. Here `L` is the exact mean colour of
each leaf of a quadtree over the source image, ANCHOR is a 2^k-entry
palette, and R1/R2 are signed per-channel corrections. With `r2_step = 1`
the three layers reconstruct `L` exactly, which is the spec's claim in §12.

## The guarantee

Every prefix of the container past the anchor decodes. The decoder reads
the header, takes whichever residual streams are fully present, and renders
from those — it never waits for a layer and never fails on a partial one.
Truncate the file anywhere and it still produces a picture at the quality
the surviving bytes pay for.

    $ head -c 40% image.nvdr > partial.nvdr
    $ nvdr_decode partial.nvdr out.ppm
    ... rendered at ANCHOR  (file carries only ANCHOR)  psnr 23.31 dB

## Build and run

    make
    ./nvdr_encode image.jpg out.nvdr
    ./nvdr_decode out.nvdr out.ppm --level 1 --compare image.jpg

`nvdr_encode` prints the cost and the PSNR of each layer, because the point
of this implementation is to test the spec's claims rather than assume
them.

## Format

    [header 32B]
    [ANCHOR : palette, geometry bitstream, packed tokens]
    [R1     : 3 planes of int8, one per channel]
    [R2     : 3 planes of int8, one per channel]

Geometry travels as one bit per quadtree node in DFS pre-order — 1 splits,
0 is a leaf — and the decoder replays the subdivision rule from the canvas
size. No coordinate is ever transmitted.

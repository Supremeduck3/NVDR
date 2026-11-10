# NVDR

A still-image and video codec written from scratch in C, with decoders
in JavaScript and WebAssembly that produce the same pixels to the bit.
No dependencies beyond libm, zlib and `stb_image.h`; nothing needs a GPU.

- **`.nvdr`** — a picture. A quadtree of leaves of 4 to 32 pixels, each
  predicted and corrected by an integer transform, arithmetic coded.
  Two modes: *progressive* (colour, then coarse texture, then fine: any
  prefix of the file is a whole picture) and *directional* (each leaf
  predicted from its decoded neighbours along 43 modes: the smaller
  file, shown whole).
- **`.nvda`** — an album: pictures in one file, each predicted from the
  one before where that pays.
- **`.nvdrv`** — a sequence: intra, P and B frames over a motion field,
  with an optional sound track.
- Any of them can ride inside an ordinary **PNG** that every viewer
  opens (`nvdr_png`).
- A picture keeps its transparency and its ICC colour profile, and a
  phone photo comes out the way up it was taken.

## Where it stands

Measured, not estimated; `NOTEBOOK.md` has every table and how it was
made. BD-rate on PSNR-Y, negative meaning NVDR needs fewer bytes:

    stills, directional, five photographs     vs AV1 (libaom)  +6.5%
                                              vs HEVC (x265)   -6.2%
    stills, progressive, five camera photos   vs JPEG          -37%
                                              vs WebP          -20%
                                              vs JPEG XL       level
                                              vs AVIF          +39%
    video, two 48-frame clips at 960x540      vs AV1 (libaom)  +19% and +25%
                                              vs HEVC (x265)   -21% and -35%
                                              vs H.264 (x264)  -42% and -52%

What that does not say: the test set is small (six thumbnails, five
photographs, a handful of short clips), the encoder optimises PSNR and
is judged by it, pictures are 8 bits a channel with no HDR, and a
sequence's sound is carried as somebody else's codec, not coded here.

Speed, on four cores: a 768x512 photograph encodes in 0.2 s progressive
and 3 s directional (in slices, one per core); the WebAssembly decoder
takes about 28 ms per megapixel, which is 38 frames a second at 720p
and 17 at 1080p.

Over a link that loses packets and cannot resend, a file in slices with
Reed-Solomon parity by layer arrives whole 80% of the time where a plain
one does 49% (5% loss in bursts), and the worst tenth of the draws show
28 dB instead of 23.

## Build and run

    make                 # the seven tools; needs gcc and zlib
    make wasm            # public/nvdr.wasm; needs clang and wasm-ld
    make check           # the regression gate
    make fuzz            # the decoders under ASan and UBSan

    ./nvdr_encode photo.jpg out.nvdr --directional --slices auto
    ./nvdr_decode out.nvdr out.png --compare photo.jpg
    ./nvdr_batch ~/Photos -r                 # a whole folder, one image per core
    ./nvdr_png out.nvdr share.png            # the same file, as a PNG
    ./nvdr_album pack album.nvda a.jpg b.jpg c.jpg
    ./nvdrv_encode frames/ out.nvdrv --fps 24 --kbps 600 --audio sound.webm
    ./nvdrv_decode out.nvdrv --out played/

    node server.js       # http://localhost:3000: encode, look, cut, play

Each tool prints its options when run without arguments.

## In a page

    <script type="module" src="nvdr-img.js"></script>
    <nvdr-img src="photo.nvdr"></nvdr-img>

`public/nvdr-img.js` decodes in a worker, with the WebAssembly decoder
when it loads and the JavaScript one otherwise, and shows a progressive
file as it arrives. `GUIA.md` lists the files to copy.

## Over a bad link

    ./nvdr_encode photo.jpg out.nvdr --slices 2
    node tools/nvdr_recv.mjs /tmp/rx 5004                    # the receiver
    node tools/nvdr_send.mjs out.nvdr 192.168.0.20 5004      # the sender

Slices keep a lost packet's damage to one band of the picture, parity
(`public/nvdr-fec.js`) fills most holes in, and the decoders take the
ranges that never arrived (`nvdr_decode --missing`, `decode()`'s
`holes`) and show the rest.

## Layout

    src/        the codec: nvdr.c (stills), nvdrv.c (sequences), nvda.c
                (albums), entropy.c, grain.c, restore.c, mc.c, pngwrap.c,
                extras.c (alpha and colour profile)
    tools/      the CLIs, and the UDP sender and receiver
    public/     the browser decoders, the <nvdr-img> element and the page
    wasm/       the C decoder built for the browser
    scripts/    the gate (verify.py), the C-against-JS cross-checks, the
                fuzzer, and analysis/, the scripts behind the notebook
    samples/    the pictures and clips the numbers were measured on
    reference/  the v9 codec, kept so its numbers stay reproducible

## Reading on

- `GUIA.md` — how to build, run and test everything, in Portuguese.
- `NOTEBOOK.md` — the measurements, in the order they were made, with
  what was tried and left out. The formats are specified there and in
  the headers: `src/nvdr.h`, `src/nvdrv.h`, `src/nvda.h`.
- `.github/workflows/ci.yml` — what runs on every push: the build, the
  gate, the decoders against each other, a short fuzz.

## Credits

The techniques are other people's, and the notebook says where each
came from: the integer transforms and angular prediction of HEVC, the
smooth and recursive-filter modes, film grain synthesis, restoration
and overlapped motion compensation of AV1, VVC's adaptive loop filter,
LZMA's range coder, Reed-Solomon erasure codes as Rizzo applied them to
packets with Cauchy matrices (Blömer et al.), and Priority Encoding
Transmission (Albanese et al.). `vendor/stb_image.h` is Sean Barrett's.

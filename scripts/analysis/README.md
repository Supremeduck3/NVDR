# Analysis tools

Throwaway measurements, kept because the numbers they produced are cited in
`../../README.md` and should be reproducible rather than trusted.

    gcc -std=c11 -O2 -I../../src -I../../vendor -o edges   edges.c   ../../src/nvdr.c ../../src/entropy.c -lm -lz
    gcc -std=c11 -O2 -I../../src -I../../vendor -o oblique oblique.c ../../src/nvdr.c ../../src/entropy.c -lm -lz

- `edges` — how many leaves stopped because they ran out of room rather
  than because they were uniform. Those are the staircase.
- `oblique` — at every node that decides to split, one oblique cut against
  the 4-way split, head to head on that node alone.

Video measurements, same rules:

    gcc -std=c11 -O2 -I../../src -I../../vendor -o frames     frames.c     ../../src/nvdr.c ../../src/entropy.c -lm -lz
    gcc -std=c11 -O2 -I../../src -I../../vendor -o interframe interframe.c ../../src/nvdr.c ../../src/entropy.c -lm -lz
    gcc -std=c11 -O2 -I../../src -I../../vendor -o residual_frame residual_frame.c ../../src/nvdr.c ../../src/entropy.c -lm -lz
    gcc -std=c11 -O2 -I../../src -I../../vendor -o chain      chain.c      ../../src/nvdr.c ../../src/entropy.c -lm -lz

- `frames` — synthesises a sequence from a still: `static`, `pan`, `zoom`,
  `object`, with `--noise N` for a stand-in sensor. Built, not filmed, so
  everything measured on it is an upper bound.
- `interframe` — how much of frame N's tree a copy-the-region rule would
  remove. The answer is "not much", and why is the interesting part.
- `residual_frame` — codes frame N as the error against frame N-1's
  reconstruction, end to end, with optional global motion search.
- `chain` — the same over a real sequence, carrying the decoder's state
  forward so accumulated drift shows up if there is any.

None of these is part of the build.

Texture layer, still images:

    gcc -std=c11 -O2 -I../../src -I../../vendor -o texture texture.c ../../src/nvdr.c ../../src/entropy.c -lm -lz

- `texture` — the rectangles cut at some level (`none`, `anchor`, `r1`,
  `r1c`, `full`), rendered flat, and what is left coded with an 8x8 DCT
  through the adaptive arithmetic coder. Real bytes, PSNR in RGB.
    gcc -std=c11 -O2 -I../../src -I../../vendor -o qtdct qtdct.c ../../src/nvdr.c ../../src/entropy.c -lm -lz

- `qtdct` — the quadtree as the DCT's partition: power-of-two leaves of 4 to
  32 px, each predicted as the mean of the decoded pixels above and to its
  left, the remainder coded with a DCT of the leaf's size, and splits
  chosen by rate and distortion against the live models.

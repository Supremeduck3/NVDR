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

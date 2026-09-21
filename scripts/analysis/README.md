# Analysis tools

Throwaway measurements, kept because the numbers they produced are cited in
`nvdr/README.md` and should be reproducible rather than trusted.

    gcc -std=c11 -O2 -I../../nvdr -I../../vendor -o edges   edges.c   ../../nvdr/nvdr.c ../../nvdr/entropy.c -lm -lz
    gcc -std=c11 -O2 -I../../nvdr -I../../vendor -o oblique oblique.c ../../nvdr/nvdr.c ../../nvdr/entropy.c -lm -lz

- `edges` — how many leaves stopped because they ran out of room rather
  than because they were uniform. Those are the staircase.
- `oblique` — at every node that decides to split, one oblique cut against
  the 4-way split, head to head on that node alone.

Neither is part of the build.

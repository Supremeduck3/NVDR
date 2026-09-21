# Frame sequences

Drop consecutive frames here, one directory per clip, numbered so they
sort in order:

    samples/video/meu_clipe/f000.png
    samples/video/meu_clipe/f001.png
    ...

PNG, JPG, BMP, TGA and PPM all read. From a video file:

    ffmpeg -i filme.mp4 -vf fps=24 -frames:v 60 f%03d.png

Every frame in a clip has to be the same size; the encoder stops at the
first one that is not.

    ../../nvdrv_encode meu_clipe/ /tmp/out.nvdrv
    ../../nvdrv_decode /tmp/out.nvdrv --compare meu_clipe/

Frames are not committed — see `.gitignore`. The numbers in `../../README.md`
under "reusing the previous frame" were measured on sequences synthesised
from stills, which have no real sensor noise, no lighting change and no
content entering at the edge, so they are optimistic. Real clips are what
they need to be re-measured against.

# Frame sequences

Getting a clip into a sandbox is the awkward part, because its outbound
access is usually restricted: in the one this was written in, GitHub
answers and every other host refuses the connection, so a link to a file
service is not a route. Pushing the zip to a branch is:

    git checkout -b frames
    git add -f clipe.zip && git commit -m "frames" && git push -u origin frames

GitHub rejects a single file over 100 MB, so a large clip goes in as
several zips or as fewer frames. Delete the branch once the frames are in.

The importer takes a zip, a folder or a URL,
finds the images however deep they are nested, renames them in reading
order, and refuses the set if the frames are not all the same size:

    python3 ../../scripts/import_frames.py clipe.zip meu_clipe
    python3 ../../scripts/import_frames.py https://host/clipe.zip meu_clipe
    python3 ../../scripts/import_frames.py /path/para/pasta meu_clipe --limit 60

Or place them by hand, one directory per clip, numbered so they sort in
order:

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

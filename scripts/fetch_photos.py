#!/usr/bin/env python3
"""
Large camera photographs for the benchmark (make bench-photos), fetched
rather than committed: they are 20-36 MB each as PNG.

    python3 scripts/fetch_photos.py        -> samples/photos/*.png

Four are raw files from rawpy's test suite, developed here with rawpy
(LibRaw: AHD demosaic, camera white balance, sRGB, 8 bits, no noise
reduction), so they were never through a lossy codec and carry the
sensor's own noise. The fifth is libjxl's test photograph, a PNG. Every
download is checked against its SHA-256; a developed PNG is kept and not
redone.

    iss030e122639     4284x2844  Nikon D3S, aurora from the ISS (NASA)
    iss042e297200     4940x3292  Nikon D4, Soyuz and aurora (NASA)
    canon_5d2         5634x3752  Canon 5D Mark II, night, high ISO
    canon_40d_sraw    1944x1296  Canon 40D sRAW, grapes in daylight
    flower            2268x1512  libjxl testdata, daylight
"""
import hashlib
import os
import sys
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'samples', 'photos')
CACHE = os.path.join(ROOT, 'output', 'photos_raw')

RAWPY = 'https://raw.githubusercontent.com/letmaik/rawpy/main/test/'
PHOTOS = [
    ('iss030e122639', RAWPY + 'iss030e122639.NEF',
     '5922721d13f11795557d97fdeb0a60b900086c402bc82a848ff280d15b99ffd4'),
    ('iss042e297200', RAWPY + 'iss042e297200.NEF',
     'b21e9752aaf8e678fe4d8138986bc9819aa89b7b969a28d43c619af3bd6d58c1'),
    ('canon_5d2', RAWPY + 'RAW_CANON_5DMARK2_PREPROD.CR2',
     '0da28f6f2718ea82fdf95c35a1790597a3a7ecbe328a3f5b60a05f4e95e380e6'),
    ('canon_40d_sraw', RAWPY + 'RAW_CANON_40D_SRAW_V103.CR2',
     '152382ce4dbf644899d12b41b4c577f07638aa3ec5745ac344c36bad93826125'),
    ('flower', 'https://raw.githubusercontent.com/libjxl/testdata/main/jxl/flower/flower.png',
     '77ea5436547c82c9be2f72308f13b64de3dc889d8c9f3708be3638e6e17cff41'),
]


def fetch(url, path, sha):
    if not os.path.exists(path):
        print(f'  baixando {url}')
        tmp = path + '.part'
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    if sha:
        got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        if got != sha:
            os.remove(path)
            sys.exit(f'{path}: SHA-256 {got}, esperado {sha}')


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    for name, url, sha in PHOTOS:
        png = os.path.join(OUT, name + '.png')
        if os.path.exists(png):
            continue
        src = os.path.join(CACHE, os.path.basename(url))
        fetch(url, src, sha)
        if src.lower().endswith('.png'):
            os.replace(src, png)
        else:
            try:
                import rawpy
                import imageio.v3 as iio
            except ImportError:
                sys.exit('revelar os raw precisa de rawpy e imageio: python3 -m pip install rawpy imageio')
            with rawpy.imread(src) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
            iio.imwrite(png, rgb)
        print(f'  {png}')


if __name__ == '__main__':
    main()

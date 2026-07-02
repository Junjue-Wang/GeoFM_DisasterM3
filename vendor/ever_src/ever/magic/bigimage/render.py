from PIL import Image
import tifffile
from tqdm import tqdm
from .sliding_window import sliding_window
import numpy as np


def render_tif(src_path, out_path, palette):
    img = tifffile.imread(src_path, out='memmap')
    rendered = tifffile.memmap(out_path, shape=tuple(list(img.shape[:2]) + [3]), dtype=np.uint8)
    for i, box in enumerate(tqdm(sliding_window(img.shape[:2], 1024, 1024))):
        xmin, ymin, xmax, ymax = box

        sub = img[ymin:ymax, xmin:xmax]

        _sub = Image.fromarray(sub)
        _sub.putpalette(palette)
        rendered[ymin:ymax, xmin:xmax] = np.asarray(_sub.convert('RGB'))

        if i % 50 == 0:
            rendered.flush()
    rendered.flush()


def extract_transparent_layer(src_path, out_path):
    img = tifffile.imread(src_path, out='memmap')
    assert img.shape[-1] == 4
    alpha = tifffile.memmap(out_path, shape=img.shape[:2], dtype=np.uint8)
    for i, box in enumerate(tqdm(sliding_window(img.shape[:2], 1024, 1024))):
        xmin, ymin, xmax, ymax = box

        alpha[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax, -1]

        if i % 50 == 0:
            alpha.flush()
    alpha.flush()


def append_transparent_layer(src_path, alpha_path, out_path):
    img = tifffile.imread(src_path, out='memmap')
    alpha = tifffile.imread(alpha_path, out='memmap')

    rendered = tifffile.memmap(out_path, shape=tuple(list(img.shape[:2]) + [4]), dtype=np.uint8)
    assert (img.shape[0] == alpha.shape[0]) and (img.shape[1] == alpha.shape[1])

    for i, box in enumerate(tqdm(sliding_window(img.shape[:2], 1024, 1024))):
        xmin, ymin, xmax, ymax = box
        rendered[ymin:ymax, xmin:xmax] = np.concatenate([img[ymin:ymax, xmin:xmax],
                                                         alpha[ymin:ymax, xmin:xmax,
                                                         None]],
                                                        axis=2)
        if i % 50 == 0:
            rendered.flush()
    rendered.flush()


def image_transform(src_path, out_path, fn):
    img = tifffile.imread(src_path, out='memmap')
    out = tifffile.memmap(out_path, shape=img.shape, dtype=np.uint8)
    for i, box in enumerate(tqdm(sliding_window(img.shape[:2], 1024, 1024))):
        xmin, ymin, xmax, ymax = box

        out[ymin:ymax, xmin:xmax] = fn(img[ymin:ymax, xmin:xmax])

        if i % 50 == 0:
            out.flush()

    out.flush()

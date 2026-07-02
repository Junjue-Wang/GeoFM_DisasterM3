import os
import glob
from skimage.io import imsave

try:
    from osgeo import gdal
except:
    pass


def tiff_split(image_path, kernel_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return os.system(
        f'gdal_retile.py -resume -co COMPRESS=LZW -ps {kernel_size} {kernel_size} -targetDir {output_dir} {image_path}')


def tiff_merge(input_dir, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    i = os.path.join(input_dir, '*.tif')
    return os.system(f'gdal_merge.py -co COMPRESS=LZW -o {output_path} {i}')

def lzw_compress_tif(src_path, dst_path):
    os.system(f'gdal_translate -co COMPRESS=LZW {src_path} {dst_path}')

def save_with_georef(fname, arr, src_path):
    imsave(fname, arr, check_contrast=False)

    content = gdal.Open(fname, gdal.GA_Update)
    original = gdal.Open(src_path, gdal.GA_Update)
    content.SetGeoTransform(original.GetGeoTransform())
    content.SetProjection(original.GetProjection())
    content.FlushCache()
    del content
    del original


def align_shape(src_dir, dst_dir):
    for out_fp in glob.glob(os.path.join(dst_dir, '*.tif')):
        t1_fp = os.path.join(src_dir, os.path.basename(out_fp))
        t1_img = gdal.Open(t1_fp)
        out = gdal.Open(out_fp)
        if t1_img.RasterXSize < out.RasterXSize or t1_img.RasterYSize < out.RasterYSize:
            out_arr = out.ReadAsArray(0, 0, t1_img.RasterXSize, t1_img.RasterYSize)
            save_with_georef(out_fp, out_arr, t1_fp)
        del t1_img
        del out

import os


def set_palette_to_tif(src_path, palette_dict: dict):
    """
    usage:
        set_palette_to_tif('./demo.tif', {0: (255, 255, 255), 1: (255, 0, 0)})
    """
    assert os.path.splitext(src_path)[1] in ['.tif', '.tiff']
    from osgeo import gdal

    color_tb = gdal.ColorTable()

    for k, v in palette_dict.items():
        color_tb.SetColorEntry(k, v)

    dataset = gdal.Open(src_path, gdal.GA_Update)

    band = dataset.GetRasterBand(1)

    band.SetColorTable(color_tb)
    # band.SetNoDataValue(0)

    band.FlushCache()
    dataset.FlushCache()
    dataset = None


def copy_color_palette(src_path, dst_path):
    from osgeo import gdal
    src = gdal.Open(src_path, gdal.GA_Update)
    band = src.GetRasterBand(1)
    color_tb = band.GetColorTable()

    dst = gdal.Open(dst_path, gdal.GA_Update)
    band = dst.GetRasterBand(1)
    band.SetColorTable(color_tb)

    band.FlushCache()
    dst.FlushCache()

    src = None
    dst = None

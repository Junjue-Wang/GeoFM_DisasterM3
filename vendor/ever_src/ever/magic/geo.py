def write_georeference(src_path, dst_path):
    from osgeo import gdal
    content = gdal.Open(dst_path, gdal.GA_Update)

    original = gdal.Open(src_path)
    content.SetGeoTransform(original.GetGeoTransform())
    content.SetProjection(original.GetProjection())

    content.FlushCache()
    content = None
    original = None

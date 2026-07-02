import tifffile
from .sliding_window import sliding_window
from tqdm import tqdm


def _index_image(image, bbox, data_format):
    xmin, ymin, xmax, ymax = bbox
    if data_format == 'HWC':
        return image[ymin:ymax, xmin:xmax]
    elif data_format == 'CHW':
        return image[:, ymin:ymax, xmin:xmax]
    else:
        raise ValueError(f'Unknown data_format: {data_format}')


def _set_value(image, bbox, value, data_format):
    xmin, ymin, xmax, ymax = bbox
    if data_format == 'HWC':
        image[ymin:ymax, xmin:xmax] = value
    elif data_format == 'CHW':
        image[:, ymin:ymax, xmin:xmax] = value
    else:
        raise ValueError(f'Unknown data_format: {data_format}')


def tiff_many2one_op(input_paths,
                     out_path,
                     custom_func,
                     data_format='HWC',
                     block_size=1024,
                     flush_interval=256,
                     out_shape=None,
                     out_dtype=None):
    imgs = [tifffile.imread(fp, out='memmap') for fp in input_paths]

    if data_format == 'HWC':
        spatial_size = imgs[0].shape[:2]
        bboxes = sliding_window(spatial_size, block_size, block_size)
    elif data_format == 'CHW':
        spatial_size = imgs[0].shape[1:]
        bboxes = sliding_window(spatial_size, block_size, block_size)
    else:
        raise ValueError(f'Unknown data_format: {data_format}')

    if out_path:
        if out_dtype is None:
            # infer data type
            region_imgs = [_index_image(im, bboxes[0], data_format) for im in imgs]
            result = custom_func(*region_imgs)
            out_dtype = result.dtype
            out_shape = [i for i in result.shape]
            if data_format == 'HWC':
                out_shape[0] = spatial_size[0]
                out_shape[1] = spatial_size[1]
            if data_format == 'CHW':
                out_shape[1] = spatial_size[0]
                out_shape[2] = spatial_size[1]
            out_shape = tuple(out_shape)

    if out_path:
        res = tifffile.memmap(out_path, shape=out_shape, dtype=out_dtype)
    for i, bbox in enumerate(tqdm(bboxes, desc='ever.tiffflow')):
        region_imgs = [_index_image(im, bbox, data_format) for im in imgs]
        region_result = custom_func(*region_imgs)
        if out_path:
            _set_value(res, bbox, region_result, data_format)
            if i % flush_interval == 0:
                res.flush()
    if out_path:
        res.flush()
        return res

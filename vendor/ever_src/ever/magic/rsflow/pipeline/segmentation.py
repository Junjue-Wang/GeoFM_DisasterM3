import glob
import os
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
from ever.core import logger
from tqdm import tqdm

from ..core import utils, gdal_utils
from ..dataset.segmentation import gen_dataloader

try:
    from osgeo import gdal
except:
    pass


def _task_on_cpu(out, output_process_fn, temp_out_dir, filenames):
    if output_process_fn:
        out = output_process_fn(out)
    assert len(out.shape) == 3, 'Tensor shape should be like (N, H, W)'
    out = out.numpy()
    for filename, o in zip(filenames, out):
        gdal_utils.save_with_georef(os.path.join(temp_out_dir, os.path.basename(filename)), o, filename)


def segmentation_on_tif(model: nn.Module,
                        image_path,
                        output_path,
                        kernel_size,
                        batch_size,
                        num_workers=0,
                        transforms=None,
                        device='cuda',
                        output_process_fn=None
                        ):
    logging = logger.get_logger()
    temp_patch_dir = os.path.join(os.curdir, '_rsflow_patch_')
    temp_out_dir = os.path.join(os.curdir, '_rsflow_tempout_')
    os.makedirs(temp_out_dir, exist_ok=True)

    logging.info('Stage 1/3: splitting image')
    state = gdal_utils.tiff_split(image_path, kernel_size=kernel_size, output_dir=temp_patch_dir)
    if state != 0:
        utils._error_handler(logging, 'Failed to run gdal_utils.tiff_split', [temp_patch_dir, temp_out_dir])
        return

    logging.info('Stage 2/3: model inference')
    if len(glob.glob(os.path.join(temp_patch_dir, '*.tif'))) > len(glob.glob(os.path.join(temp_out_dir, '*.tif'))):
        ppe = ProcessPoolExecutor(max_workers=2 * num_workers)
        dataloader = gen_dataloader(temp_patch_dir, batch_size, num_workers, transforms)
        th_device = torch.device(device)
        model.to(th_device)
        model.eval()
        with torch.no_grad():
            for img, filenames in tqdm(dataloader, desc='rsflow.segmentation'):
                if utils._is_exists([os.path.join(temp_out_dir, os.path.basename(fp)) for fp in filenames]):
                    continue
                img = img.to(th_device)
                out = model(img)
                ppe.submit(_task_on_cpu, out.cpu(), output_process_fn, temp_out_dir, filenames)
        ppe.shutdown()
    else:
        logging.info('Find cached results. Skip Stage 2/3: model inference')
    logging.info('Stage 3/3: merging results')
    gdal_utils.align_shape(temp_patch_dir, temp_out_dir)
    state = gdal_utils.tiff_merge(temp_out_dir, output_path)
    if state != 0:
        utils._error_handler(logging, 'Failed to run gdal_utils.tiff_merge', [])
        return

import glob
import os
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
from ever.core import logger
from tqdm import tqdm

from ..core import utils, gdal_utils
from ..dataset.pairwise_change import gen_dataloader

try:
    from osgeo import gdal
except:
    pass


def _pairwise_split(logging, t1_image_path, t2_image_path, kernel_size):
    temp_t1_patch_dir = os.path.join(os.curdir, '_rsflow_t1_patch_')
    temp_t2_patch_dir = os.path.join(os.curdir, '_rsflow_t2_patch_')
    temp_out_dir = os.path.join(os.curdir, '_rsflow_tempout_')
    os.makedirs(temp_out_dir, exist_ok=True)

    logging.info('Stage 1/3: splitting image')
    state = gdal_utils.tiff_split(t1_image_path, kernel_size=kernel_size, output_dir=temp_t1_patch_dir)
    if state != 0:
        utils._error_handler(logging, 'Failed to run t1 gdal_utils.tiff_split',
                             [temp_t1_patch_dir])
        return
    state = gdal_utils.tiff_split(t2_image_path, kernel_size=kernel_size, output_dir=temp_t2_patch_dir)
    if state != 0:
        utils._error_handler(logging, 'Failed to run t2 gdal_utils.tiff_split',
                             [temp_t2_patch_dir])
        return

    return temp_t1_patch_dir, temp_t2_patch_dir, temp_out_dir


def _task_on_cpu_for_bcd(out, output_process_fn, temp_out_dir, t1_filenames, t2_filenames):
    if output_process_fn:
        out = output_process_fn(out)
    assert len(out.shape) == 3, 'Tensor shape should be like (N, H, W)'
    out = out.numpy()
    for t1_filename, t2_filename, o in zip(t1_filenames, t2_filenames, out):
        gdal_utils.save_with_georef(os.path.join(temp_out_dir, os.path.basename(t1_filename)), o,
                                    t1_filename)


def pairwise_binary_change_detection_on_tif(model: nn.Module,
                                            t1_image_path,
                                            t2_image_path,
                                            output_path,
                                            kernel_size,
                                            batch_size,
                                            num_workers=0,
                                            transforms=None,
                                            device='cuda',
                                            output_process_fn=None
                                            ):
    logging = logger.get_logger()
    temp_t1_patch_dir, temp_t2_patch_dir, temp_out_dir = _pairwise_split(logging, t1_image_path, t2_image_path,
                                                                         kernel_size)

    logging.info('Stage 2/3: model inference')
    if len(glob.glob(os.path.join(temp_t1_patch_dir, '*.tif'))) > len(glob.glob(os.path.join(temp_out_dir, '*.tif'))):
        ppe = ProcessPoolExecutor()
        dataloader = gen_dataloader(temp_t1_patch_dir, temp_t2_patch_dir, batch_size, num_workers, transforms)
        th_device = torch.device(device)
        model.to(th_device)
        model.eval()
        with torch.no_grad():
            for img, t1_filenames, t2_filenames in tqdm(dataloader, desc='rsflow.pairwise_change'):
                if utils._is_exists([os.path.join(temp_out_dir, os.path.basename(fp)) for fp in t1_filenames]):
                    continue
                img = img.to(th_device)
                out = model(img)
                ppe.submit(_task_on_cpu_for_bcd, out.cpu(), output_process_fn, temp_out_dir, t1_filenames, t2_filenames)
        ppe.shutdown()
    else:
        logging.info('Find cached results. Skip Stage 2/3: model inference')

    logging.info('Stage 3/3: merging results')
    gdal_utils.align_shape(temp_t1_patch_dir, temp_out_dir)
    state = gdal_utils.tiff_merge(temp_out_dir, output_path)
    if state != 0:
        utils._error_handler(logging, 'Failed to run gdal_utils.tiff_merge',
                             [])
        return
    utils._rm_temp([temp_t1_patch_dir, temp_t2_patch_dir])


def _task_on_cpu_for_scd(out, _output_process_fn, _temp_out_dir, _t1_filenames, _t2_filenames):
    if _output_process_fn:
        _out = _output_process_fn(out)
    assert len(_out.shape) == 4, 'Tensor shape should be like (N, C, H, W)'
    assert _out.size(1) == 3, 'The number of channel should be 3'
    _out = _out.permute(0, 2, 3, 1).numpy()  # NCHW->NHWC
    for t1_filename, t2_filename, o in zip(_t1_filenames, _t2_filenames, _out):
        gdal_utils.save_with_georef(os.path.join(_temp_out_dir, os.path.basename(t1_filename)), o,
                                    t1_filename)


def pairwise_semantic_change_detection_on_tif(model: nn.Module,
                                              t1_image_path,
                                              t2_image_path,
                                              output_path,
                                              kernel_size,
                                              batch_size,
                                              num_workers=0,
                                              transforms=None,
                                              device='cuda',
                                              output_process_fn=None
                                              ):
    logging = logger.get_logger()
    temp_t1_patch_dir, temp_t2_patch_dir, temp_out_dir = _pairwise_split(logging,
                                                                         t1_image_path,
                                                                         t2_image_path,
                                                                         kernel_size)

    logging.info('Stage 2/3: model inference')
    if len(glob.glob(os.path.join(temp_t1_patch_dir, '*.tif'))) > len(glob.glob(os.path.join(temp_out_dir, '*.tif'))):
        ppe = ProcessPoolExecutor(max_workers=2 * num_workers)
        dataloader = gen_dataloader(temp_t1_patch_dir, temp_t2_patch_dir, batch_size, num_workers, transforms)
        th_device = torch.device(device)
        model.to(th_device)
        model.eval()
        with torch.no_grad():
            for img, t1_filenames, t2_filenames in tqdm(dataloader, desc='rsflow.pairwise_change'):
                if utils._is_exists([os.path.join(temp_out_dir, os.path.basename(fp)) for fp in t1_filenames]):
                    continue
                img = img.to(th_device)
                out = model(img)
                ppe.submit(_task_on_cpu_for_scd, out.cpu(), output_process_fn, temp_out_dir, t1_filenames, t2_filenames)
        ppe.shutdown()
    else:
        logging.info('Find cached results. Skip Stage 2/3: model inference')

    logging.info('Stage 3/3: merging results')
    gdal_utils.align_shape(temp_t1_patch_dir, temp_out_dir)
    state = gdal_utils.tiff_merge(temp_out_dir, output_path)
    if state != 0:
        utils._error_handler(logging, 'Failed to run gdal_utils.tiff_merge',
                             [])
        return

    utils._rm_temp([temp_t1_patch_dir, temp_t2_patch_dir])

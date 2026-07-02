from skimage import measure
import numpy as np
from shapely.geometry import Polygon


def mask_to_poly(mask):
    yxs = measure.find_contours(mask, 0.5)
    xy = np.flip(yxs[0], axis=1)
    return tuple([(e[0], e[1]) for e in xy])


def poly_to_wkt(poly_xy, tolerance=0.5):
    return Polygon(poly_xy).simplify(tolerance=tolerance).wkt


def proposal(probability_mask, threshold=0.5, min_size=0):
    instance_mask, num_ins = measure.label(probability_mask > threshold, return_num=True, connectivity=2)
    MINSIZE = min_size

    poly_wkt_list = []
    score_list = []
    for idx in range(num_ins):
        ins_id = idx + 1
        bin_ins_mask = instance_mask == ins_id
        score = np.sum(probability_mask * bin_ins_mask.astype(np.float32)) / np.sum(bin_ins_mask)
        if np.sum(bin_ins_mask) > MINSIZE:
            poly_xy = mask_to_poly(bin_ins_mask)
            if len(poly_xy) >= 3:
                wkt_polygon = poly_to_wkt(poly_xy)
                poly_wkt_list.append(wkt_polygon)
                score_list.append(score)
    return poly_wkt_list, score_list

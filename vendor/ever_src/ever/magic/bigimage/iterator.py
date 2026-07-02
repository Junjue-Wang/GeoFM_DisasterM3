from .sliding_window import sliding_window


def iterator(image, kernel_size, stride):
    for box in sliding_window(image.shape[:2], kernel_size, stride):
        xmin, ymin, xmax, ymax = box
        yield image[ymin:ymax, xmin:xmax], box

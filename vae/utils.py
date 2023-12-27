import cv2
import numpy as np
import dask.array as da


def log_banner(log_function, msg):
    """Call log_function with a blank line, msg, and an underline."""
    log_function("")
    log_function("-" * len(msg))
    log_function(msg)
    log_function("-" * len(msg))


def log_multiline(log_function, msg):
    """Call log_function once for each line of msg."""
    for line in msg.split("\n"):
        log_function(line)


def log_transform(img_batch):
    """Log-transform uint16 image patch pixel values."""
    
    img_batch = np.log10(img_batch.astype('float32') + 1)
    
    return img_batch


def compute_vignette_mask(img_batch, kernel_size=40):
    """Compute a 2D Gaussian-distributed vignette mask to apply to image patches."""
    kernel_y = cv2.getGaussianKernel(img_batch.shape[1], kernel_size).astype('float32') 
    kernel_x = cv2.getGaussianKernel(img_batch.shape[2], kernel_size).astype('float32') 
    kernel = kernel_y * kernel_x.T

    # normalize kernel to be between 0 and 1
    mask = cv2.normalize(kernel, None, 0, 1, cv2.NORM_MINMAX)
    mask = mask[np.newaxis, :, :, np.newaxis]  # adding a third dimension to mask (i.e. X, X, 1)

    return mask


def clip_outlier_pixels(img_batch, percentile_cutoffs):
    """Clip lower and upper percentile outliers to 0 or 1, respectively
    based on pixel intensities of whole image."""
    pc = np.array(list(percentile_cutoffs.values()), np.float32)
    pc_min = pc[:, 0]
    pc_max = pc[:, 1]
    img_batch = np.clip((img_batch - pc_min) / (pc_max - pc_min), 0, 1)

    return img_batch


def transposeZarr(z):
    """rearrange Zarr dimensions to fit shape of expected VAE input
    (i.e. cells, x, y, channels). Returns a Dask array."""
    z = da.from_zarr(z).transpose([1, 2, 3, 0])

    return z

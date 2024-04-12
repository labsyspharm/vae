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


def transposeZarr(z):
    """rearrange Zarr dimensions to fit shape of expected VAE input
    (i.e. cells, x, y, channels). Returns a Dask array."""
    z = da.from_zarr(z).transpose([1, 2, 3, 0])

    return z


def log_transform(img_batch):
    """Log-transform uint16 image patch pixel values."""
    
    img_batch = np.log10(img_batch.astype('float32') + 1)
    
    return img_batch


def clip_outlier_pixels(img_batch, percentile_cutoffs):
    """Clip lower and upper percentile outliers to 0 or 1, respectively
    based on pixel intensities of whole image."""
    
    pc = np.array(list(percentile_cutoffs.values()), np.float32)
    pc_min = pc[:, 0]
    pc_max = pc[:, 1]
    img_batch = np.clip((img_batch - pc_min) / (pc_max - pc_min), 0, 1)

    return img_batch


def compute_vignette_mask(img_batch, std_dev):
    """Compute a 2D Gaussian-distributed vignette mask to apply to image patches."""

    window_size = img_batch.shape[1]
    
    # create a range spanning 3 STDs below and above a mean value of zero
    x = np.linspace(0 - 3 * std_dev, 0 + 3 * std_dev, 100)  
    
    # find the number of intervals (pixels) needed to account for 3 STDs below and above the mean
    num_pixels = int(x.max() - x.min())
    
    # ensure num_pixels is odd so there are an equal number of pixels
    # to the left/right, top-bottom of center pixel after cropping the mask
    if num_pixels % 2 == 1:
        pass  # number is already odd
    else:
        num_pixels + 1

    # create Gaussian kernel
    if num_pixels > window_size:
        kernel = cv2.getGaussianKernel(num_pixels, std_dev).astype('float32')
    else:
        kernel = cv2.getGaussianKernel(window_size, std_dev).astype('float32')
    
    # create 2D mask
    mask = (kernel * kernel.T)  
    
    # normalize mask pixel values 0-1
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX)

    # add additional dimensions to mask to match img_batch (i.e. cell, X, Y, channel)
    mask = mask[np.newaxis, :, :, np.newaxis] 
    
    # store min and max intensity values before mask is cropped for later visualization
    vmin = mask.min()
    vmax = mask.max()
    
    def crop_vignette_mask(mask, window_size):
    
        # determine the center of the larger mask
        center = np.array(mask.shape) // 2
        
        # calculate half-size
        half_size = window_size // 2
        
        # define the range of indices for cropping
        start_x = max(0, center[1] - half_size)
        end_x = min(mask.shape[1], center[1] + half_size)
        start_y = max(0, center[2] - half_size)
        end_y = min(mask.shape[2], center[2] + half_size)
        
        # crop the larger array into the smaller mask
        cropped_mask = mask[:, start_x:end_x, start_y:end_y, :]

        # return the smaller mask
        return cropped_mask
    
    if num_pixels > window_size:

        # crop the mask to desired cellcutter window size
        mask = crop_vignette_mask(mask=mask, window_size=window_size)
    
    return mask, vmin, vmax

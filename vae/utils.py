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

    # for multi-tissue analysis
    # extract percentile cutoffs for all labels in batch
    # pc_min = np.array(
    #     [list(percentile_cutoffs[label].values()) for label in labels], np.float32
    # )[:, :, 0] 
    # pc_max = np.array(
    #     [list(percentile_cutoffs[label].values()) for label in labels], np.float32
    # )[:, :, 1]
    
    # # clip and normalize each image in the batch on a global per channel basis 
    # img_batch = np.clip(
    #     (img_batch - pc_min[:, None, None, :]) / 
    #     (pc_max[:, None, None, :] - pc_min[:, None, None, :]), 0, 1)
    
    # for single tissue analysis
    pc = np.array(list(percentile_cutoffs.values()), np.float32)
    pc_min = pc[:, 0]
    pc_max = pc[:, 1]
    img_batch = np.clip((img_batch - pc_min) / (pc_max - pc_min), 0, 1)

    return img_batch


def apply_scaling(flat_patch, scaling_funcs, sample, sample_keys):
    for j, key in enumerate(sample_keys):
        scaling_function = scaling_funcs[key][('gmm_peak1', 'mp99.95')]
        flat_patch[:, j] = scaling_function(flat_patch[:, j])
    return flat_patch


def align_histograms(X_block, labels_block, limits):
    """Apply linear polynomial image channel transformations computed 
   in align_histograms.py to a batch of image patches block-wise and clip
   channels to ensure 0-1 normalization."""

    # flat_patches = X_block.reshape(X_block.shape[0], -1, X_block.shape[-1])
    # for i, sample in enumerate(labels_block):
    #     func_keys = [k for k in scaling_funcs.keys() 
    #                  if k[0] == sample.item()]
    #     # assuming scaling_funcs keys are in same order as patch channels 
    #     for j, key in enumerate(func_keys):
    #         scaling_function = scaling_funcs[key][('gmm_peak1', 'mp99.95')]
    #         flat_patches[i, :, j] = scaling_function(flat_patches[i, :, j])
    # X_block_transformed = flat_patches.reshape(X_block.shape)
    # return np.clip(X_block_transformed, 0, 1)
    
    X_block = X_block.astype('float32')
    sample_keys = {sample: [] for sample in np.unique(labels_block.ravel())}
    for key in limits.keys():
        if key[0] in sample_keys.keys():
            sample_keys[key[0]].append(key)
    num_samples, _, _, num_channels = X_block.shape
    mins = np.empty((num_samples, num_channels), dtype='float32')
    maxs = np.empty((num_samples, num_channels), dtype='float32')
    for i, sample in enumerate(labels_block.ravel()):
        for j, key in enumerate(sample_keys[sample]):
            min_val = limits[key]
            mins[i, j] = min_val
            maxs[i, j] = 65535
    mins = mins.reshape(num_samples, 1, 1, num_channels)
    maxs = maxs.reshape(num_samples, 1, 1, num_channels)
    X_block = log_transform(X_block)
    log_mins = np.log10(mins + 1)
    log_maxs = np.log10(maxs + 1)
    range_vals = log_maxs - log_mins
    mask = X_block > log_mins
    scaled_values = (X_block - log_mins) / range_vals
    X_block = np.where(mask, scaled_values, 0)
    X_block = np.clip(X_block, 0, 1)
    return X_block


def compute_vignette_mask(window_size, std_dev):
    """Compute a 2D Gaussian-distributed vignette mask to apply to image patches."""
    
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


def reverse_processing(percentile_cutoffs, channel_slice, channel_name, contrast_limits):
    """Reverses percentile normalization and log10-transformation,
       pixel outliers remained clipped)."""

    lower_cutoff_log, upper_cutoff_log = percentile_cutoffs[channel_name]

    # reverse percentile normalization
    channel_slice = (
        (((upper_cutoff_log - lower_cutoff_log) * (channel_slice - 0)) /
         (1 - 0)) + lower_cutoff_log
    )

    # reverse log10-transform
    channel_slice = np.rint(10 ** channel_slice)

    # Normalize pixel values between lower and upper percentile bounds
    # lower = np.rint(10**lower_cutoff_log)
    # upper = np.rint(10**upper_cutoff_log)
    # channel_slice = (channel_slice-lower) / (upper-lower)

    # Apply image contrast settings
    lower = contrast_limits[channel_name][0]
    upper = contrast_limits[channel_name][1]
    channel_slice = (channel_slice - lower) / (upper - lower)

    channel_slice = np.clip(channel_slice, 0, 1)

    return channel_slice

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


def remove_background(X_block, samples_block, bkgd_limits):
    """Set channel background pixels to zero in a batch of image patches.
    This function clips the pixel values of each channel to ensure normalization within the range of 0 to 1. 
    It utilizes background limits to adjust the pixel intensities based on the specified transformations 
    and handles different data types (uint8 and uint16) accordingly.

    Args:
        X_block (ndarray): A batch of image patches with shape (num_samples, height, width, num_channels).
        labels_block (ndarray): An array of sample labels corresponding to each image patch.
        bkgd_limits (dict): A dictionary containing background limits for each sample.

    Returns:
        ndarray: The transformed and normalized image patches.
    """

    # extract patch metadata
    if str(X_block.dtype) == 'uint8':
        divisor = 255
    elif str(X_block.dtype) == 'uint16':
        divisor = 65535
    else:
        raise ValueError(f'Unsupported data type: {X_block.dtype}')
    num_samples, _, _, num_channels = X_block.shape    
    
    # convert patch data to float32
    X_block = X_block.astype('float32')
    
    # compute channel mins and maxs
    channel_mins = np.array([
        bkgd_limits[key] for key in bkgd_limits.keys() for
        i in samples_block.ravel() if i == key[0]
    ]).reshape(num_samples, num_channels)
    channel_maxs = np.full((num_samples, num_channels), divisor, dtype='float32')

    # log transform patches and channel mins/maxs
    X_block = log_transform(X_block)
    log_channel_mins = np.log10(channel_mins + 1).reshape(num_samples, 1, 1, num_channels)
    log_channel_maxs = np.log10(channel_maxs + 1).reshape(num_samples, 1, 1, num_channels)
    range_vals = log_channel_maxs - log_channel_mins
    
    # mask background pixels and normalize channel intensities
    foreground_mask = X_block > log_channel_mins
    normalized_values = (X_block - log_channel_mins) / range_vals
    X_block = np.where(foreground_mask, normalized_values, 0)
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
        num_pixels += 1

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
        end_x = start_x + window_size  # Ensure the width is window_size
        start_y = max(0, center[2] - half_size)
        end_y = start_y + window_size  # Ensure the height is window_size

        # start_x = max(0, center[1] - half_size)
        # end_x = min(mask.shape[1], center[1] + half_size)
        # start_y = max(0, center[2] - half_size)
        # end_y = min(mask.shape[2], center[2] + half_size)
        
        # crop the larger array into the smaller mask
        cropped_mask = mask[:, start_x:end_x, start_y:end_y, :]

        # return the smaller mask
        return cropped_mask
    
    if num_pixels > window_size:

        # crop the mask to desired cellcutter window size
        mask = crop_vignette_mask(mask=mask, window_size=window_size)
    
    return mask, vmin, vmax


def reverse_processing(X_decoded_block, X_block, samples_block, bkgd_limits, contrast_limits, mask):
    """Reverse image patch processing operations on reconstructed (decoded) image patches"""

    # extract patch metadata
    if str(X_block.dtype) == 'uint8':
        divisor = 255
    elif str(X_block.dtype) == 'uint16':
        divisor = 65535
    else:
        raise ValueError(f'Unsupported data type: {X_block.dtype}')
    num_samples, _, _, num_channels = X_block.shape

    # get log-transformed channel mins
    channel_mins = np.array([
        bkgd_limits[key] for key in bkgd_limits.keys() for
        i in samples_block.ravel() if i == key[0]
    ]).reshape(num_samples, num_channels)

    log_channel_mins = np.log10(channel_mins + 1).reshape(num_samples, 1, 1, num_channels)

    # get log-transformed channel maxs
    channel_maxs = np.full((num_samples, num_channels), divisor, dtype='float32')
    log_channel_maxs = np.log10(channel_maxs + 1).reshape(num_samples, 1, 1, num_channels)
    
    range_vals = log_channel_maxs - log_channel_mins

    # mask background pixels of original channel slice image
    log_original = log_transform(X_block) 
    foreground_mask = log_original > log_channel_mins

    if mask is not None:
        reversed_block = X_decoded_block / mask
    else:
        reversed_block = X_decoded_block

    # reverse log transformation and scaling
    reversed_block = np.where(foreground_mask, reversed_block * range_vals + log_channel_mins, 0)
    reversed_block = np.rint(10 ** reversed_block) - 1

    # apply image contrast settings to reverse-transformed channel slice
    lower = np.array(
        [i[0] for i in contrast_limits.values()] * samples_block.shape[0]
    ).reshape(num_samples, 1, 1, num_channels)
    upper = np.array(
        [i[1] for i in contrast_limits.values()] * samples_block.shape[0]
    ).reshape(num_samples, 1, 1, num_channels)

    reversed_block = (reversed_block - lower) / (upper - lower)

    # clip values to 0-1 range
    reversed_block = np.clip(reversed_block, 0, 1)

    return reversed_block


def num_legend_columns(bbox, ax, legend_elements, size=10):
    """calculate the number of columns to use in the legend given
       the number of legend entries and the height of the 
       target bbox y-axis"""
    
    num_legend_entries = len(legend_elements)
    x_axis_width, y_axis_height = bbox.width, bbox.height

    # determine the maximum number of entries per column based on y-axis height
    denominator = 0.4 # adjust denominator based on your legend entry height
    max_entries_per_column = int(y_axis_height / denominator)  

    # create multiple columns for the legend if necessary
    if num_legend_entries > max_entries_per_column:
        num_columns = (num_legend_entries // max_entries_per_column) + 1
    else:
        num_columns = 1
    
    ax.legend(
        handles=legend_elements, prop={'size': size}, labelspacing=0.5, 
        bbox_to_anchor=(1.01, 1.0), ncol=num_columns, columnspacing=0.3
    )

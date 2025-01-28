import math
import pickle

import numpy as np


def save(dataset, file):
    
    with open(file, 'wb') as fo:
        
        pickle.dump(dataset, fo)


def unpickle(file):
    
    with open(file, 'rb') as fo:
        
        dict = pickle.load(fo, encoding='bytes')
    
    return dict


def batch_run(function, images, batch_size=5000):
    '''
    function   : lambda function taking images with shape [N,H,W,C] as input
    images     : tensor of shape [N,H,W,C]
    batch_size : batch size
    '''
    
    res = []
    
    for i in range(math.ceil(len(images) / batch_size)):
        
        res.append(function(images[i*batch_size:(i+1)*batch_size]))
    
    return np.concatenate(res, axis=0)


def preprocess(attrs, q1, q2, use_abs=False):
    
    if use_abs:
        attrs = np.abs(attrs)

    # identify percentile thresholds
    attrs_thresh_low = np.percentile(attrs, q1, axis=(1, 2, 3), keepdims=True)
    attrs_thresh_high = np.percentile(attrs, q2, axis=(1, 2, 3), keepdims=True)

    # create boolean mask with same shape as attrs; 
    # filled with True if attrs_thresh_low is positive
    # filled with False if attrs_thresh_low is negative
    pos = np.tile(
        attrs_thresh_low > 0, 
        [1, attrs.shape[1],
         attrs.shape[2],
         attrs.shape[3]
         ]
        )

    # get indices where attribution values are less than attrs_thresh_low 
    ind = np.where(attrs < attrs_thresh_low)
    
    # clip array to low and upper threshold values
    attrs_clip = np.clip(attrs, attrs_thresh_low, attrs_thresh_high)

    # set attribution values less than attrs_thresh_low to zero
    # if attrs_thresh_low is positive
    attrs_clip[ind] = (1 - pos[ind]) * attrs_clip[ind]

    return attrs_clip, attrs_thresh_low, attrs_thresh_high


def pixel_range(img):
    vmin, vmax = np.min(img), np.max(img)

    if vmin * vmax >= 0:
        
        v = np.maximum(np.abs(vmin), np.abs(vmax))
        
        return [-v, v], 'bwr'
    
    else:

        if -vmin > vmax:
            vmax = -vmin
        else:
            vmin = -vmax

        return [vmin, vmax], 'bwr'


def scale(x):
    
    return x / 127.5 - 1.0

import os
import pickle
import logging

import numpy as np
import pandas as pd

from tifffile import imread

import matplotlib.pyplot as plt

from ..utils import log_banner, log_multiline, log_transform

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


def MAKE_FEATURE_PROCESSING_SELECTIONS(config):

    save_dir = os.path.join(config.output_path, '4_feature_preprocessing_selections')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        markers = pd.read_csv(config.markers_path)

        # format plot grid
        numRows = 4
        numColumns = 8
        grid_dims = (numRows, numColumns)

        # initialize figure canvas
        fig_orig = plt.figure(figsize=(12, 8.5))
        fig_log = plt.figure(figsize=(12, 8.5))
        fig_clip = plt.figure(figsize=(12, 8.5))

        # loop over cellcutter channels
        cutoffs = {}
        for e, marker in enumerate(config.tif_channels):

            print(marker)

            # get channel number from markers.csv
            channel_number = markers['channel_number'][markers['marker_name'] == marker].values[0]

            # read channel
            img = imread(config.tif_path, key=channel_number - 1)
            
            # log-transform image
            log_img = log_transform(img)

            # ignore zeros when computing lower and upper percentile cutoffs 
            non_zero = log_img[log_img > 0]

            # specify lower and upper percentile cutoffs
            lower_cutoff_log = np.percentile(non_zero.ravel(), config.cutoffs[0])
            upper_cutoff_log = np.percentile(non_zero.ravel(), config.cutoffs[1])

            # add channel cutoffs to dict
            cutoffs[marker] = (lower_cutoff_log, upper_cutoff_log)

            # scale 0.17th and 99.99th percentile between 0 and 1
            # Note: this will cause outlier pixels below the 0.17th percentile
            # and above the 99.99th to take values <0 and >1, respectively.
            # Then clip outliers to lower and upper percentile cutoffs (i.e., 0-1)
            clip_rescaled_log_img = np.clip(
                (log_img - lower_cutoff_log) / (upper_cutoff_log - lower_cutoff_log), 0, 1
            )

            # add channel subplot to figures
            ax_orig = fig_orig.add_subplot(grid_dims[0], grid_dims[1], e + 1)
            ax_log = fig_log.add_subplot(grid_dims[0], grid_dims[1], e + 1)
            ax_clip = fig_clip.add_subplot(grid_dims[0], grid_dims[1], e + 1)

            # plot original channel histogram
            vals, bins, patches = ax_orig.hist(
                img.ravel(), bins=60, color='tab:blue', alpha=0.7, rwidth=0.85
            )
            ax_orig.title.set_text(marker)

            # plot log-transformed channel histogram
            vals, bins, patches = ax_log.hist(
                log_img.ravel(), bins=60, color='tab:blue', alpha=0.7, rwidth=0.85
            )
            ax_log.vlines(
                x=[np.percentile(non_zero.ravel(), config.cutoffs[0]),
                   np.percentile(non_zero.ravel(), config.cutoffs[1])],
                ymin=0, ymax=vals.max(), color='tab:red'
            )
            ax_log.title.set_text(marker)

            # plot normalized channel histogram
            vals, bins, patches = ax_clip.hist(
                clip_rescaled_log_img.ravel(), bins=60, color='tab:blue', alpha=0.7, rwidth=0.85
            )
            ax_clip.title.set_text(marker)

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99, hspace=0.2)
        plt.tight_layout()
        fig_orig.savefig(os.path.join(save_dir, 'log_hists_orig.pdf'))
        fig_log.savefig(os.path.join(save_dir, 'log_hists_log.pdf'))
        fig_clip.savefig(os.path.join(save_dir, 'log_hists_clip.pdf'))
        plt.close('all')

        # save cutoffs
        with open(os.path.join(save_dir, 'cutoffs.pkl'), 'wb') as handle:
            pickle.dump(cutoffs, handle, protocol=pickle.HIGHEST_PROTOCOL)

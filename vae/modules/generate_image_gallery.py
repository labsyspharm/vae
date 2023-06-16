import logging

import os
import math
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from skimage.color import gray2rgb
from skimage.util import img_as_float

import zarr

from ..utils import log_banner, log_multiline

logger = logging.getLogger(__name__)
# log_multiline(logger.info, pd.DataFrame().to_string(index=False))


def PlotInputImgs(numExamples, numColumns, imgs, labels, fontSize, colors, channelNames, channelIDs, fileName, contrast_limits):

    numSamples = len(imgs)
    numRows = math.ceil(numExamples/numColumns)
    grid_dims = (numRows, numColumns)

    # numColumns = math.ceil(numExamples/numRows)
    # grid_dims = (numRows, numColumns)

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(13, 10))

    custom_lines = []
    for e, row in enumerate(labels.iterrows()):

        plt.subplot(grid_dims[0], grid_dims[1], e + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # initialize array of zeros with shape of full-size image
        overlay = np.zeros((imgs.shape[2], imgs.shape[3]))

        overlay = gray2rgb(overlay)

        for d, ch, color in zip(channelIDs, channelNames, colors):

            lyr = imgs[d, row[0], :, :]

            lyr = img_as_float(lyr)

            # apply image contrast settings
            lyr -= (contrast_limits[ch][0]/65535)
            lyr /= (
                (contrast_limits[ch][1]/65535)
                - (contrast_limits[ch][0]/65535))

            lyr = np.clip(lyr, 0, 1)

            lyr = gray2rgb(lyr)
            lyr = lyr * color
            overlay += lyr

            custom_lines.append(Line2D([0], [0], color=color, lw=5))

        label = row[1]['cluster']

        plt.imshow(overlay, cmap=plt.cm.binary)
        plt.xlabel(label, size=fontSize, labelpad=1.5)

    fig.legend(
        custom_lines, channelNames, prop={'size': 11},
        bbox_to_anchor=(0.98, 0.99)
        )

    plt.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.85)
    plt.savefig(os.path.join(save_dir, f'{fileName}.pdf'))
    plt.close('all')


def GENERATE_IMAGE_GALLERY(config):

    cellcutter_input_path = os.path.join(
        config.output_path, '1_cellcutter_input'
        )

    save_dir = os.path.join(config.output_path, '3_thumbnail_examples')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        viz_marker_ids = [
            cellcutter_markers.index(i) for i in config.viz_channels
            ]

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # read training labels
        labels_path = os.path.join(cellcutter_input_path, 'train.csv')
        labels = pd.read_csv(labels_path)

        # read training images
        z_path = os.path.join(
            cellcutter_output_path,
            f'train_thumbnails_{config.window_size}.zarr'
            )
        store = zarr.ZipStore(z_path, mode='r')
        z = zarr.open(store=store)

        # contrast settings
        contrast_limits = yaml.safe_load(open(config.contrast_path))

        # pull random thumbnails from training data to check quality
        thumb_ids = np.random.RandomState(1).choice(
            range(0, z.shape[1]), num_examples, replace=False)

        imgs = z.get_orthogonal_selection((slice(None), thumb_ids))

        labels = labels.iloc[thumb_ids]
        labels.reset_index(drop=True, inplace=True)
        labels.sort_values(by='cluster', inplace=True)

        colors = plt.get_cmap('tab10').colors * math.ceil(imgs.shape[3]/10)

        PlotInputImgs(
            numExamples=num_examples,
            numColumns=16,
            imgs=imgs,
            labels=labels,
            fontSize=8,
            colors=colors,
            channelNames=viz_markers,
            channelIDs=viz_marker_ids,
            fileName='thumbnail_examples',
            contrast_limits=contrast_limits
            )

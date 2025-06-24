import os
import sys
import yaml
import logging

import numpy as np
import pandas as pd

import math
from datetime import datetime

from natsort import natsorted
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from matplotlib import colors
from matplotlib.path import Path
from matplotlib.colors import to_rgb
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
from matplotlib.colors import ListedColormap
from matplotlib.widgets import LassoSelector
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from skimage.color import gray2rgb
from skimage.util import img_as_float

import zarr
import dask.array as da

from keras.models import load_model

import hdbscan
from umap import UMAP
from sklearn.manifold import TSNE

from joblib import Memory

from ..utils import (
    log_banner, log_multiline, log_transform, 
    remove_background, compute_vignette_mask, 
    transposeZarr, reverse_processing, num_legend_columns
)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


def categorical_cmap(numUniqueSamples, numCatagories, cmap='tab10', continuous=False):

    numSubcatagories = math.ceil(numUniqueSamples / numCatagories)

    if numCatagories > plt.get_cmap(cmap).N:
        raise ValueError('Too many categories for colormap.')
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, numCatagories))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(numCatagories, dtype=int))
        
        # rearrange hue order to taste
        cd = {
            'B': 0, 'O': 1, 'G': 2, 'R': 3, 'Pu': 4,
            'Br': 5, 'Pi': 6, 'Gr': 7, 'Y': 8, 'Cy': 9,
        }
        myorder = [
            cd['B'], cd['O'], cd['G'], cd['Pu'], cd['Y'],
            cd['R'], cd['Cy'], cd['Br'], cd['Gr'], cd['Pi']
        ]
        ccolors = [ccolors[i] for i in myorder]
        
        # regular palette
        # use Okabe and Ito color-safe palette for first 6 colors
        ccolors[0] = np.array([0.91, 0.29, 0.235])  # E84A3C
        ccolors[1] = np.array([0.18, 0.16, 0.15])  # 2E2926
        ccolors[0] = np.array([0.0, 0.447, 0.698, 1.0])  # blue
        ccolors[1] = np.array([0.902, 0.624, 0.0, 1.0])  # orange
        ccolors[2] = np.array([0.0, 0.620, 0.451, 1.0])  # bluish green
        ccolors[3] = np.array([0.8, 0.475, 0.655, 1.0])  # reddish purple
        ccolors[4] = np.array([0.941, 0.894, 0.259, 1.0])  # yellow
        ccolors[5] = np.array([0.835, 0.369, 0.0, 1.0])  # vermillion

    cols = np.zeros((numCatagories * numSubcatagories, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, numSubcatagories).reshape(numSubcatagories, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, numSubcatagories)
        arhsv[:, 2] = np.linspace(chsv[2], 1, numSubcatagories)
        rgb = colors.hsv_to_rgb(arhsv)
        cols[i * numSubcatagories:(i + 1) * numSubcatagories, :] = rgb
    cmap = colors.ListedColormap(cols)

    # trim colors if necessary
    if len(cmap.colors) > numUniqueSamples:
        trim = len(cmap.colors) - numUniqueSamples
        cmap_colors = cmap.colors[:-trim]
        cmap = colors.ListedColormap(cmap_colors, name='from_list', N=None)

    return cmap


def ScatterReconstructions(X_decoded, X_encoded_embedded, zoom, ax):

    def imscatter(x, y, imageData, ax, zoom):

        images = []
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            # clip values to 0-1 range (avoids matplotlib clipping warning)
            img = np.clip(imageData[i], 0, 1)
            image = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(
                image, (x0, y0), xycoords='data', frameon=False)
            images.append(ax.add_artist(ab))

        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
    
    imscatter(
        x=X_encoded_embedded[:, 0], y=X_encoded_embedded[:, 1],
        imageData=X_decoded, ax=ax, zoom=zoom)


def PlotLatentSpace(reconstructions, zoom, X_encoded_embedded, X_decoded_reversed, y, channel_color_dict, scatter_point_size, filename, save_dir):

    fig, ax = plt.subplots(figsize=(10, 10))

    if reconstructions:
        
        ScatterReconstructions(
            X_decoded=X_decoded_reversed, X_encoded_embedded=X_encoded_embedded, zoom=zoom, ax=ax
        )

        legend_elements = []
        for name, color in channel_color_dict.items():
            legend_elements.append(Line2D([0], [0], color=color, lw=6, label=name))

        ax.scatter(
            X_encoded_embedded[:, 0], X_encoded_embedded[:, 1], 
            c='k', s=0.0, ec='k', lw=0.25, zorder=4
        )
        
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        num_legend_columns(bbox=bbox, ax=ax, legend_elements=legend_elements)

        ax.set_aspect('equal', adjustable='box')
        plt.grid(False)
        
        plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=600, bbox_inches='tight')
        plt.close('all')

    else:

        cmap = categorical_cmap(
            numUniqueSamples=len(y.unique()), 
            numCatagories=10, cmap='tab10', continuous=False
        )

        label_color_dict = dict(
            zip(natsorted(y.unique()), 
                [tuple(i) for i in cmap.colors])
        )

        if -1 in y.unique():
            # make black the first color to specify
            # cluster outliers (i.e. cluster -1 cells)
            cmap = ListedColormap(
                np.insert(
                    arr=cmap.colors, obj=0, values=[0.0, 0.0, 0.0], axis=0)
            )

            # trim qualitative cmap to number of unique samples
            cmap = ListedColormap(cmap.colors[:-1])

        hue_dict = dict(
            zip(natsorted(y.unique()), 
                list(range(len(y.unique()))))
        )

        c = [hue_dict[i] for i in y]

        ax.scatter(
            X_encoded_embedded[:, 0], X_encoded_embedded[:, 1], 
            c=c, cmap=cmap, ec='k', lw=0.0, s=scatter_point_size
        )

        legend_elements = []
        for e, i in enumerate(natsorted(y.unique())):

            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', label=i,
                       markerfacecolor=cmap.colors[e], 
                       markeredgecolor=None, lw=0.25, markersize=9)
            )

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        num_legend_columns(bbox=bbox, ax=ax, legend_elements=legend_elements)

        ax.set_aspect('equal', adjustable='box')
        plt.grid(False)
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=600, bbox_inches='tight')
        plt.close('all')

        return label_color_dict


def DecodeVectors(decoder, X_encoded, X, X_seg, sample_labels, bkgd_limits, contrast_limits, channel_color_dict, tif_channels, patch_dims, mask, chunk_size, intensity_multiplier):
    
    # decode latent vector
    print('Decoding images...')
    X_decoded = decoder.predict(X_encoded, batch_size=200)
    X_decoded = da.from_array(
        X_decoded, chunks=(chunk_size, patch_dims[0], patch_dims[1], patch_dims[2])
    )

    X_decoded_reversed = da.map_blocks(
        reverse_processing, X_decoded, X, sample_labels,
        bkgd_limits, contrast_limits, mask,
        dtype=np.float32  
        # dtype required to avoid ValueError: dtype inference failed in map_blocks
    )  # call .compute() to debug
    
    # slice out channels to visualize
    channel_indices = np.array([tif_channels.index(i) for i in channel_color_dict.keys()])
    X_decoded_reversed = X_decoded_reversed[:, :, :, channel_indices]
    
    # convert to RGB, brighten, and colorize
    X_decoded_reversed = gray2rgb(X_decoded_reversed)
    X_decoded_reversed *= intensity_multiplier
    color_arr = np.array(
        [to_rgb(color) for _, color in channel_color_dict.items()]
    ).reshape(1, 1, 1, -1, 3)
    X_decoded_reversed *= color_arr

    # add segmentation outlines layer
    X_seg = img_as_float(X_seg)
    seg_slices_rgb = gray2rgb(X_seg) * 0.25  # decrease alpha
    X_decoded_reversed = np.concatenate((X_decoded_reversed, seg_slices_rgb), axis=3)

    # add centroid layer
    centroid_layer = np.zeros(
        (X_decoded_reversed.shape[0], X_decoded_reversed.shape[1],
         X_decoded_reversed.shape[2], 1, 3)
    )
    centroid_layer[
        :, int(X_decoded_reversed.shape[1] / 2),
        int(X_decoded_reversed.shape[2] / 2), 0, :
    ] = 1
    X_decoded_reversed = np.concatenate((X_decoded_reversed, centroid_layer), axis=3)
    
    # sum images along channels axis to generate final RGB image patches
    X_decoded_reversed = np.sum(X_decoded_reversed, axis=3)

    # rechunk reverse processed image patches
    X_decoded_reversed = da.rechunk(
        X_decoded_reversed, 
        chunks=(X_decoded_reversed.chunksize[0], 
                X_decoded_reversed.chunksize[1],
                X_decoded_reversed.chunksize[2],
                3)
    )

    return X_decoded, X_decoded_reversed


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def LassoVectors(contrast_limits, patch_dims, imgs_instead_of_points, zoom, X, X_seg, X_encoded, X_encoded_embedded, X_decoded_reversed, y, numColumns, intensity_multiplier, max_examples, tif_channels, channel_color_dict, patch_font_size, save_dir):

    subplot_kw = dict(
        xlim=(X_encoded_embedded[:, 0].min(), X_encoded_embedded[:, 0].max()),
        ylim=(X_encoded_embedded[:, 1].min(), X_encoded_embedded[:, 1].max()),
        autoscale_on=False
    )

    fig, lasso_ax = plt.subplots(subplot_kw=subplot_kw, figsize=(9, 8))

    if imgs_instead_of_points is True:

        ScatterReconstructions(
            X_decoded=X_decoded_reversed, X_encoded_embedded=X_encoded_embedded, 
            zoom=zoom, ax=lasso_ax
        )

        legend_elements = []
        for name, color in channel_color_dict.items():

            legend_elements.append(Line2D([0], [0], color=color, lw=5, label=name))

        pts = lasso_ax.scatter(
            X_encoded_embedded[:, 0], X_encoded_embedded[:, 1], c='k',
            s=0.0, ec='k', lw=0.25, zorder=4
        )

    else:
        cmap = categorical_cmap(
            numUniqueSamples=len(y.unique()), 
            numCatagories=10, cmap='tab10', continuous=False
        )

        if -1 in y.unique():
            # make black the first color to specify
            # cluster outliers (i.e. cluster -1 cells)
            cmap = ListedColormap(
                np.insert(
                    arr=cmap.colors, obj=0, values=[0.0, 0.0, 0.0], axis=0)
            )

            # trim qualitative cmap to number of unique samples
            cmap = ListedColormap(cmap.colors[:-1])

        hue_dict = dict(
            zip(natsorted(y.unique()), 
                [tuple(i) for i in cmap.colors])
        )

        c = [hue_dict[i] for i in y]

        pts = lasso_ax.scatter(
            X_encoded_embedded[:, 0], X_encoded_embedded[:, 1],
            c=c, s=30.0, ec='k', lw=0.25, zorder=4
        )

        lasso_ax.update_datalim(
            np.column_stack([X_encoded_embedded[:, 0], X_encoded_embedded[:, 1]])
        )
        lasso_ax.autoscale()

        legend_elements = []
        for e, i in enumerate(natsorted(y.unique())):

            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', label=i,
                       markerfacecolor=cmap.colors[e], 
                       markeredgecolor=None, lw=0.25, markersize=9)
            )
    
    bbox = lasso_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    num_legend_columns(bbox=bbox, ax=lasso_ax, legend_elements=legend_elements)

    plt.tight_layout()

    selector = SelectFromCollection(lasso_ax, pts)

    def accept(event):
        if event.key == "enter":
            # print("Selected points:")
            # print(selector.xys[selector.ind])
            selector.disconnect()
            lasso_ax.set_title("")
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    lasso_ax.set_title("Press enter to accept selected points, then close window.")
    lasso_ax.set_aspect('equal')
    plt.show(block=True)

    y.reset_index(drop=True, inplace=True)
    selected_labels = pd.DataFrame(data={'label': y.loc[selector.ind]})
    if len(selected_labels) > max_examples:
        selected_labels = selected_labels.sample(n=max_examples, random_state=44)

    # sort selected labels
    selected_labels.sort_values(by='label', inplace=True)
    # selected_labels.sort_index(inplace=True)  # sort by row index for efficient indexing

    # isolated encodings and image patches associated with lasso selection
    X_encoded = X_encoded[selected_labels.index]
    X = X[selected_labels.index]
    X_seg = X_seg[selected_labels.index]
    X_decoded_reversed = X_decoded_reversed[selected_labels.index]

    # check cell images
    numSamples = len(selected_labels)
    numRows = math.ceil(numSamples / numColumns)
    grid_dims = (numRows, numColumns)

    fig = plt.figure()

    fig.text(0.13, 0.97, 'Input Images', ha='left', fontsize='medium')
    fig.text(0.53, 0.97, 'Learned Reconstructions', fontsize='medium')

    outer_grid_rows = 1
    outer_grid_cols = 2

    outer = gridspec.GridSpec(
        outer_grid_rows, outer_grid_cols, wspace=0.1, hspace=0.0)

    for panel in range(outer_grid_rows * outer_grid_cols):

        inner = gridspec.GridSpecFromSubplotSpec(
            grid_dims[0], grid_dims[1],
            subplot_spec=outer[panel], wspace=0.1, hspace=0.0)

        for e, (_, label) in enumerate(selected_labels.iterrows()):

            ax = plt.Subplot(fig, inner[e])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if panel == 0:

                input_img = X[e]

                # apply image contrast settings to reverse-transformed channel slice
                lower = np.array(
                    [i[0] for i in contrast_limits.values()]).reshape(1, 1, input_img.shape[2])
                upper = np.array(
                    [i[1] for i in contrast_limits.values()]).reshape(1, 1, input_img.shape[2])

                input_img = (input_img - lower) / (upper - lower)

                # slice out channels to visualize
                channel_indices = np.array(
                    [tif_channels.index(i) for i in channel_color_dict.keys()]
                )
                input_img = input_img[:, :, channel_indices]

                # convert to RGB, brighten, and colorize
                input_img = gray2rgb(input_img)
                input_img *= intensity_multiplier
                color_arr = np.array(
                    [to_rgb(color) for _, color in channel_color_dict.items()]
                ).reshape(1, 1, -1, 3)
                input_img *= color_arr

                # add segmentation outlines layer
                seg_layer = X_seg[e]
                seg_layer = img_as_float(seg_layer)
                seg_layer = gray2rgb(seg_layer) * 0.25  # decrease alpha
                input_img = np.concatenate((input_img, seg_layer), axis=2)

                # # add centroid layer
                centroid_layer = np.zeros(
                    (input_img.shape[0], input_img.shape[1], 1, 3)
                )
                centroid_layer[
                    int(input_img.shape[0] / 2),
                    int(input_img.shape[1] / 2), 0, :
                ] = 1
                input_img = np.concatenate((input_img, centroid_layer), axis=2)
    
                # sum images along channels axis to generate final RGB image patches
                overlay = np.sum(input_img, axis=2)

            elif panel == 1:

                overlay = X_decoded_reversed[e]

            # clip values to 0-1 range (avoids matplotlib clipping warning)
            overlay = np.clip(overlay, 0, 1)
            
            ax.imshow(overlay)

            # ax.set_xlabel(label['label'], fontsize=patch_font_size, labelpad=0.75)
            fig.add_subplot(ax)

    fig.subplots_adjust(bottom=0.01, top=0.94, left=0.01, right=0.85, wspace=0.2, hspace=0.1)

    legend_elements = []
    for name, color in channel_color_dict.items():
        legend_elements.append(Line2D([0], [0], color=color, lw=3, label=name))

    bbox = transforms.Bbox.from_extents(0, 0, 0, fig.get_size_inches()[1])
    num_legend_columns(bbox=bbox, ax=fig, legend_elements=legend_elements, size=5)

    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'lassoed_cells.png'), dpi=800, bbox_inches='tight')
    plt.close('all')


def PlotReconstructedImages(patch_dims, X, y, X_seg, X_decoded_reversed, contrast_limits, numColumns, tif_channels, channel_color_dict, intensity_multiplier, patch_font_size, filename, save_dir):

    selected_labels = pd.DataFrame(data={'label': y})

    # sort selected labels
    selected_labels.sort_values(by='label', inplace=True)
    # selected_labels.sort_index(inplace=True)  # sort by row index for efficient indexing

    selected_labels.reset_index(drop=True, inplace=True)

    # isolated encodings and image patches associated with lasso selection
    X = X[selected_labels.index]
    X_seg = X_seg[selected_labels.index]
    X_decoded_reversed = X_decoded_reversed[selected_labels.index]

    numSamples = len(X)
    numRows = math.ceil(numSamples / numColumns)
    grid_dims = (numRows, numColumns)

    fig = plt.figure()

    fig.text(0.13, 0.97, 'Input Images', ha='left', fontsize='medium')
    fig.text(0.53, 0.97, 'Learned Reconstructions', ha='left', fontsize='medium')

    outer_grid_rows = 1
    outer_grid_cols = 2

    outer = gridspec.GridSpec(outer_grid_rows, outer_grid_cols, wspace=0.1, hspace=0.0)

    for panel in range(outer_grid_rows * outer_grid_cols):

        inner = gridspec.GridSpecFromSubplotSpec(
            grid_dims[0], grid_dims[1],
            subplot_spec=outer[panel], wspace=0.1, hspace=0.0
        )

        for e, (_, label) in enumerate(selected_labels.iterrows()):

            ax = plt.Subplot(fig, inner[e])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if panel == 0:

                input_img = X[e]

                # apply image contrast settings
                lower = np.array(
                    [i[0] for i in contrast_limits.values()]).reshape(1, 1, input_img.shape[2])
                upper = np.array(
                    [i[1] for i in contrast_limits.values()]).reshape(1, 1, input_img.shape[2])

                input_img = (input_img - lower) / (upper - lower)

                # slice out channels to visualize
                channel_indices = np.array(
                    [tif_channels.index(i) for i in channel_color_dict.keys()]
                )
                input_img = input_img[:, :, channel_indices]

                # convert to RGB, brighten, and colorize
                input_img = gray2rgb(input_img)
                input_img *= intensity_multiplier
                color_arr = np.array(
                    [to_rgb(color) for _, color in channel_color_dict.items()]
                ).reshape(1, 1, -1, 3)
                input_img *= color_arr

                # add segmentation outlines layer
                seg_layer = X_seg[e]
                seg_layer = img_as_float(seg_layer)
                seg_layer = gray2rgb(seg_layer) * 0.25  # decrease alpha
                input_img = np.concatenate((input_img, seg_layer), axis=2)

                # # add centroid layer
                centroid_layer = np.zeros(
                    (input_img.shape[0], input_img.shape[1], 1, 3)
                )
                centroid_layer[
                    int(input_img.shape[0] / 2),
                    int(input_img.shape[1] / 2), 0, :
                ] = 1
                input_img = np.concatenate((input_img, centroid_layer), axis=2)
    
                # sum images along channels axis to generate final RGB image patches
                overlay = np.sum(input_img, axis=2)

            elif panel == 1:

                overlay = X_decoded_reversed[e]

            # clip values to 0-1 range (avoids matplotlib clipping warning)
            overlay = np.clip(overlay, 0, 1)
            
            ax.imshow(overlay)

            # ax.set_xlabel(label['label'], fontsize=patch_font_size, labelpad=0.75)
            fig.add_subplot(ax)

    fig.subplots_adjust(bottom=0.01, top=0.94, left=0.01, right=0.85, wspace=0.2, hspace=0.1)

    legend_elements = []
    for name, color in channel_color_dict.items():
        legend_elements.append(Line2D([0], [0], color=color, lw=3, label=name))

    bbox = transforms.Bbox.from_extents(0, 0, 0, fig.get_size_inches()[1])
    num_legend_columns(bbox=bbox, ax=fig, legend_elements=legend_elements, size=5)

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=800, bbox_inches='tight')
    plt.close('all')


def mse(patch_dims, X_transform, y, X_seg, X_decoded, X_decoded_reversed, mse_percentile_cutoff, filename, save_dir):

    errors = []

    for transformed_img, reconstructed_img in zip(X_transform, X_decoded):

        err = np.sum((transformed_img.compute() - reconstructed_img) ** 2)
        err /= float(transformed_img.shape[0] * transformed_img.shape[1])

        errors.append(err)

    average_error = np.mean(errors)
    print(f'average mean squared error is {average_error}')

    n, bins, pathes = plt.hist(errors, bins=50)
    plt.axvline(np.percentile(errors, mse_percentile_cutoff), c='r')
    plt.xlabel('Mean squared error')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=800)

    threshold = np.percentile(errors, mse_percentile_cutoff)
    outlier_idxs = [
        i for i, v in enumerate(errors) if v > threshold
    ]

    X_transform_outliers = X_transform[outlier_idxs]
    y_outliers = y[outlier_idxs].reset_index(drop=True)
    X_outliers_seg = X_seg[outlier_idxs]
    X_decoded_reversed_outliers = X_decoded_reversed[outlier_idxs]

    plt.close('all')

    return (average_error, errors, X_transform_outliers, y_outliers, X_outliers_seg,
            X_decoded_reversed, outlier_idxs, X_decoded_reversed_outliers)


def InterpolationGrid(patch_dims, grid_size, X_encoded, y, decoder, label_color_dict, channel_color_dict, frac_of_scatter_points, scatter_point_size, make_sample_sizes_equal, img_brightness_multiplier, scatter_point_alpha, save_dir):

    # make lists to store grid coordinates and their indices
    # for every latent space dimension
    grids = []
    indices = []

    # grab dimensions in reverse order (e.g. 3, 2, 1, 0) with grids.reverse()
    for d in range(2):

        # round minimum latent variable in dimension 'd' down to 100th place
        flr = math.floor(X_encoded[:, d].min() * 100.0) / 100.0

        # round maximum latent variable in dimension 'd' up to 100th place
        cel = math.ceil(X_encoded[:, d].max() * 100.0) / 100.0

        # construct grid of latent dimension values and their grid indices
        grid = np.array(np.linspace(flr, cel, grid_size))
        grids.append(grid)

        idx = np.array(range(0, len(grid)))
        idx = idx.astype(int)
        indices.append(idx)

    grids.reverse()

    # create an empty array that will fit the required number of
    # patches given the chosen grid size
    y_dim, x_dim, channels = (patch_dims[0], patch_dims[1], patch_dims[2])
    figure = np.zeros((y_dim * grid_size, x_dim * grid_size))

    # convert to RGB
    figure = gray2rgb(figure)

    # sample the grid
    for grid_tup, dim_tup in zip(product(*grids), product(*indices)):

        grid_tup = list(grid_tup)
        grid_tup.reverse()

        # get z sample at current grid spec
        z_sample = np.array([grid_tup])

        # decode z sample
        X_decoded = decoder.predict(z_sample)

        # reconstruct image
        reconstructed_img = X_decoded.reshape(
            y_dim, x_dim, channels)

        # create blank image to append channels to
        overlay = np.zeros((reconstructed_img.shape[0], reconstructed_img.shape[1]))

        # convert to RGB
        overlay = gray2rgb(overlay)

        # append chanenels
        for name, (ch, color) in channel_color_dict.items():

            channel = reconstructed_img[:, :, ch]

            channel = gray2rgb(channel)

            channel = channel * img_brightness_multiplier

            overlay += channel * color

        figure[dim_tup[0] * y_dim: (dim_tup[0] + 1) * y_dim,
               dim_tup[1] * x_dim: (dim_tup[1] + 1) * x_dim] = overlay

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(figure)
    plt.grid(linestyle='dotted', linewidth=0.0)
    plt.gca().invert_yaxis()

    ax.set_xticks(list(range(x_dim, grid_size * x_dim + 1, x_dim)))
    ax.set_xticklabels(list(np.round(grids[1], 2)), size=8)
    ax.set_yticks(list(range(y_dim, grid_size * y_dim + 1, y_dim)))
    ax.set_yticklabels(list(np.round(grids[0], 2)), size=8)

    y = y.reset_index(drop=True)  # ensure y and X_encoded indices match
    y = y.sample(frac=frac_of_scatter_points)  # sample y
    data = X_encoded[y.index]  # get corresponding X_encoded samples
    y = y.reset_index(drop=True)  # reset y index to matche sampled X_encoded

    scatter_df = pd.concat([pd.DataFrame(y), pd.DataFrame(data)], axis=1)

    if make_sample_sizes_equal is True:
        lengths_list = []
        for i in scatter_df['cluster_3d'].unique():

            lengths_list.append(len(scatter_df[scatter_df['cluster_3d'] == i]))

        sample_size = min(lengths_list)

        sample_dfs = []
        for j in scatter_df['cluster_3d'].unique():
            if len(scatter_df[scatter_df['cluster_3d'] == j]) != sample_size:
                sample_dfs.append(scatter_df[scatter_df['cluster_3d'] == j].sample(n=sample_size))
            else:
                sample_dfs.append(scatter_df[scatter_df['cluster_3d'] == j])

        scatter_df = pd.concat(sample_dfs, axis=0)
    else:
        pass

    # get global x and y ranges
    global_x_min = scatter_df[0].min()
    global_x_max = scatter_df[0].max()
    global_y_min = scatter_df[1].min()
    global_y_max = scatter_df[1].max()

    data = data[scatter_df.index]
    scatter_df = scatter_df.reset_index(drop=True)

    # filter latent vectors to isolate those between
    # the latent variable ranges used to generate the sweep grid
    scatter_points = scatter_df[
        (scatter_df[0] > grids[1].min())
        & (scatter_df[0] < grids[1].max())
        & (scatter_df[1] > grids[0].min())
        & (scatter_df[1] < grids[0].max())
    ].copy()

    # map latent space units (percent point function)
    # to the x, y pixel ranges of the sweep grid
    scatter_points[0] = np.interp(
        scatter_points[0],
        (global_x_min, global_x_max),
        (patch_dims[0] / 2, (figure.shape[0] - patch_dims[0] / 2))
    )
    scatter_points[1] = np.interp(
        scatter_points[1],
        (global_y_min, global_y_max),
        (patch_dims[0] / 2, (figure.shape[0] - patch_dims[0] / 2))
    )

    # plot latent vectors for images
    for i in natsorted(scatter_points['cluster_3d'].unique()):

        ax.scatter(
            scatter_points[0][scatter_points['cluster_3d'] == i],
            scatter_points[1][scatter_points['cluster_3d'] == i],
            fc=[
                label_color_dict[i] for i in scatter_points['cluster_3d'][
                    scatter_points['cluster_3d'] == i]],
            marker='o', label=i, s=scatter_point_size, ec='k', lw=0.25, alpha=scatter_point_alpha
        )

    # channel legend
    legend_elements = []
    for name, (ch, color) in channel_color_dict.items():

        legend_elements.append(Line2D([0], [0], lw=6, color=color, label=name))

    leg = plt.legend(
        handles=legend_elements, loc='upper left', prop={'size': 11},
        markerscale=1, labelspacing=0.7, bbox_to_anchor=(1, 0.32)
    )
    ax.add_artist(leg)

    # cluster legend
    ax.legend(
        loc='upper left', prop={'size': 11}, labelspacing=0.6,
        markerscale=3, bbox_to_anchor=(1, 1.0))

    plt.xticks(rotation=90)

    plt.xlabel('latent dimension 1', size=13, labelpad=10, fontweight='normal')
    plt.ylabel('latent dimension 2', size=13, labelpad=10, fontweight='normal')

    plt.savefig(os.path.join(save_dir, 'InterpolationGrid.png'), dpi=800, bbox_inches='tight')
    plt.close('all')

    return global_x_min, global_x_max, global_y_min, global_y_max, scatter_df


def ENCODE_IMAGES(config):

    if not os.path.isfile(
       os.path.join(config.output_path, 'checkpoints/ENCODE_IMAGES.txt')):
        
        ###########################################################################
        # I/O
        
        rng = np.random.default_rng(237)
        
        save_dir = os.path.join(
            config.output_path, f'7_latent_space_LD{config.latent_dimension}'
        )
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        patch_dims = (
            config.window_size, config.window_size, len(config.tif_channels)
        )

        # read background limits computed in remove_background.py
        bkgd_limits = yaml.safe_load(
            open(os.path.join(config.output_path, 
                              '5_background_limits/bkgd_limits.yml'))
        )
        bkgd_limits = {eval(k): v for (k, v) in bkgd_limits.items()}

        contrast_limits = yaml.safe_load(open(config.contrast_path))
        
        # ensure keys in config.tif_channel order
        contrast_limits = {k: contrast_limits[k] for k in config.tif_channels}  
        
        # load previously saved encoder and decoder
        try:
            encoder = load_model(
                os.path.join(config.output_path, '6_train_vae/encoder.hdf5')
            )
        except OSError:
            print('Encoder not found.')
            sys.exit()

        try:
            decoder = load_model(
                os.path.join(config.output_path, '6_train_vae/decoder.hdf5')
            )
        except OSError:
            print('Decoder not found.')
            sys.exit()

        ###########################################################################
        # generate combined training, validation, and test image patch zarrs
        
        chunk_size = 200
        
        combo_dir = os.path.join(save_dir, 'combined_zarr')
        combo_dir_seg = os.path.join(save_dir, 'combined_zarr_seg')
        
        if not os.path.exists(combo_dir):
            os.makedirs(combo_dir)

            print('Combined data does not exist, creating...')
            print()
            
            # read training patches
            z1_train_path = os.path.join(
                config.output_path,
                f'3_cellcutter_output_win{config.window_size}/'
                f'train_patches_{config.window_size}_qc.zip'
            )
            store = zarr.ZipStore(z1_train_path, mode='r')
            X_train = zarr.open(store=store)

            # read validation patches
            z1_validate_path = os.path.join(
                config.output_path,
                f'3_cellcutter_output_win{config.window_size}'
                f'/validate_patches_{config.window_size}_qc.zip'
            )
            store = zarr.ZipStore(z1_validate_path, mode='r')
            X_validate = zarr.open(store=store)

            # read test patches
            z1_test_path = os.path.join(
                config.output_path,
                f'3_cellcutter_output_win{config.window_size}'
                f'/test_patches_{config.window_size}_qc.zip'
            )
            store = zarr.ZipStore(z1_test_path, mode='r')
            X_test = zarr.open(store=store)

            # initialize combo zarr to store combined training, validation, and test data
            # (cellcutter cuts at 1 cell per chunk for efficient indexing during model training
            # with shuffling, which requires random-access indexing. Rechunking the number of cells
            # per chunk here for efficient image patch encoding.)
            X_combo = zarr.open(
                combo_dir,
                mode='w',
                shape=(X_train.shape[0], X_train.shape[1], X_train.shape[2], 
                       X_train.shape[3]),
                chunks=(X_train.chunks[0], chunk_size, X_train.chunks[2], 
                        X_train.chunks[3]),
                compressor=X_train.compressor, dtype=X_train.dtype
            )
            
            X_combo[:] = X_train
            X_combo.append(X_validate, axis=1)
            X_combo.append(X_test, axis=1)

        if not os.path.exists(combo_dir_seg):
            os.makedirs(combo_dir_seg)

            print('Combined segmentation outlines does not exist, creating...')
            print()
            
            # read cell outlines patches for training data
            z1_train_path_seg = os.path.join(
                config.output_path,
                f'3_cellcutter_output_win{config.window_size}'
                f'/train_patches_{config.window_size}_qc_seg.zip'
            )
            store = zarr.ZipStore(z1_train_path_seg, mode='r')
            X_train_seg = zarr.open(store=store)

            # read cell outlines patches for validation data
            z1_validate_path_seg = os.path.join(
                config.output_path,
                f'3_cellcutter_output_win{config.window_size}'
                f'/validate_patches_{config.window_size}_qc_seg.zip'
            )
            store = zarr.ZipStore(z1_validate_path_seg, mode='r')
            X_validate_seg = zarr.open(store=store)

            # read cell outlines patches for test data
            z1_test_path_seg = os.path.join(
                config.output_path,
                f'3_cellcutter_output_win{config.window_size}'
                f'/test_patches_{config.window_size}_qc_seg.zip'
            )
            store = zarr.ZipStore(z1_test_path_seg, mode='r')
            X_test_seg = zarr.open(store=store)

            # initialize combo zarr to store combined training, validation, and test data
            X_combo_seg = zarr.open(
                combo_dir_seg,
                mode='w',
                shape=(
                    X_train_seg.shape[0], X_train_seg.shape[1], 
                    X_train_seg.shape[2], X_train_seg.shape[3]
                ),
                chunks=(
                    X_train_seg.chunks[0], chunk_size, X_train_seg.chunks[2],
                    X_train_seg.chunks[3]
                ),
                compressor=X_train_seg.compressor, dtype=X_train_seg.dtype
            )
            
            X_combo_seg[:] = X_train_seg
            X_combo_seg.append(X_validate_seg, axis=1)
            X_combo_seg.append(X_test_seg, axis=1)
        
        ###########################################################################
        # sample image patch data

        # read combined patches
        X = zarr.open(combo_dir)
        X = transposeZarr(z=X)
        chunk0 = X.chunksize[0]

        # read combined patches
        X_seg = zarr.open(combo_dir_seg)
        X_seg = transposeZarr(z=X_seg)

        # read training labels
        y_train = pd.read_csv(
            os.path.join(config.output_path, '1_cellcutter_input/train_qc.csv')
        )

        # read validation labels
        y_validate = pd.read_csv(
            os.path.join(config.output_path, '1_cellcutter_input/validate_qc.csv')
        )

        # read test labels
        y_test = pd.read_csv(
            os.path.join(config.output_path, '1_cellcutter_input/test_qc.csv')
        )

        # combine labels for training, validation, and test data
        y = pd.concat([y_train, y_validate, y_test], axis=0)
        y['Sample'] = y['Sample'].astype(str)

        if not config.cluster_full_dataset:
            
            idxs = rng.choice(X.shape[0], size=config.clustering_sample_size, replace=False)
            
            # slicing can change the original chunk size, ensure equivalent chunk sizes between
            # filtered X and sample_labels in da.map_blocks by 
            # explicitly rechunking axis0 (number of cells)
            chunk0 = X.chunksize[0]
            X = X[idxs].rechunk({0: chunk0})
            X_seg = X_seg[idxs].rechunk({0: chunk0})
            
            y = y.iloc[idxs].copy()

        ###########################################################################
        # preprocess image patch data
        
        mask, vmin, vmax = compute_vignette_mask(
            window_size=config.window_size, std_dev=config.mask_std_dev
        )

        # X_transform = X.astype('float')  # use for binary patches

        sample_labels = da.from_array(y['Sample'].values, chunks=(chunk0))
        sample_labels = sample_labels.reshape((-1, 1, 1, 1))

        print('Log transforming image patches and removing background')
        print()

        X_transform = da.map_blocks(
            remove_background, X, sample_labels, bkgd_limits, 
            dtype=np.float32  
            # dtype required to avoid ValueError: dtype inference failed in map_blocks
        )

        if config.masked_model:
            print('Applying Gaussian vignette mask')
            print()
            X_transform *= mask

        ###########################################################################
        # encode images
        
        if config.cluster_full_dataset:
            num_cells = 'FULL'
        else:
            num_cells = X_transform.shape[0]
        
        try:
            X_encoded = np.load(
                os.path.join(save_dir, f'encodings_{num_cells}.npy')
            )

        except (FileNotFoundError):
            print('Encoding images...')

            X_encoded = encoder.predict(X_transform, batch_size=chunk_size)

            np.save(
                os.path.join(
                    save_dir, f'encodings_{num_cells}'), X_encoded
            )

            temp = pd.DataFrame(X_encoded)
            temp.columns = [f'vae_{i}' for i in temp.columns]
            x_encoded_df = pd.concat(
                [y['CellID'].reset_index(drop=True), pd.DataFrame(temp)], axis=1)
            x_encoded_df.to_csv(
                os.path.join(
                    save_dir, 
                    f'encodings_{num_cells}.csv'), 
                index=False
            )

        ###########################################################################
        # embed latent vectors
        
        if (config.latent_dimension > 2):
            embedding_path = os.path.join(save_dir, f'embedding_{num_cells}.npy')
            
            try:
                # load previously saved embedding
                X_encoded_embedded = np.load(embedding_path)

            except (FileNotFoundError):

                startTime = datetime.now()

                print('Embedding data...')

                if config.embedding_algorithm == 'TSNE':
                    print('Computing TSNE embedding.')
                    X_encoded_embedded = TSNE(
                        n_components=2,
                        perplexity=27,
                        early_exaggeration=19,
                        learning_rate=200.0,
                        metric='euclidean',
                        random_state=5,
                        init='pca', n_jobs=-1).fit_transform(X_encoded)
                elif config.embedding_algorithm == 'UMAP':
                    print('Computing UMAP embedding.')

                    X_encoded_embedded = UMAP(
                        n_components=2,
                        n_neighbors=25,
                        learning_rate=1.0,
                        output_metric='euclidean',
                        min_dist=0.1,
                        repulsion_strength=3,
                        random_state=1,
                        n_epochs=1000,
                        init='spectral',
                        metric='euclidean',
                        metric_kwds=None,
                        output_metric_kwds=None,
                        n_jobs=-1,
                        low_memory=False,
                        spread=1.0,
                        local_connectivity=1.0,
                        set_op_mix_ratio=0.5,
                        negative_sample_rate=5,
                        transform_queue_size=4.0,
                        a=None,
                        b=None,
                        angular_rp_forest=False,
                        target_n_neighbors=-1,
                        target_metric='categorical',
                        target_metric_kwds=None,
                        target_weight=0.5,
                        transform_seed=42,
                        transform_mode='embedding',
                        force_approximation_algorithm=False,
                        verbose=False,
                        unique=False,
                        densmap=False,
                        dens_lambda=2.0,
                        dens_frac=0.6,
                        dens_var_shift=0.1,
                        disconnection_distance=None,
                        output_dens=False).fit_transform(X_encoded)

                    print('Embedding completed in ' + str(datetime.now() - startTime))

                # save embedding
                np.save(os.path.join(save_dir, f'embedding_{num_cells}'), X_encoded_embedded)

        else:
            # simply assign the 2D X_encoded the variable X_encoded_embedded
            X_encoded_embedded = X_encoded.copy()

        ###########################################################################
        # cluster VAE encodings in embedding space with HDBSCAN
        
        print(f'Minimum_cluster_size is {config.hdbscan_min_cluster_size}')

        clustering = hdbscan.HDBSCAN(
            min_cluster_size=config.hdbscan_min_cluster_size, min_samples=None,
            metric='euclidean', alpha=1.0, p=None, algorithm='best',
            leaf_size=40,
            memory=Memory(location=None),
            approx_min_span_tree=True,
            gen_min_span_tree=False, core_dist_n_jobs=4,
            cluster_selection_method='eom',
            allow_single_cluster=False,
            prediction_data=False,
            match_reference_implementation=False).fit(X_encoded_embedded)
        print(np.unique(clustering.labels_))
        print()

        ###########################################################################
        # plot VAE encodings in embedding space colored by various labels
        
        labels_list = {
            'cylinter': y[config.cluster_column],
            'hdbscan': pd.Series(clustering.labels_), 
            'sample': y['Sample'],
            'condition': y['Condition']
        }
        
        try:
            leiden_clusters = pd.read_csv(
                os.path.join(save_dir, f'encodings_{num_cells}-patches.csv')
            )

            if not config.cluster_full_dataset:
                leiden_clusters = leiden_clusters.iloc[idxs]
            
            labels_list['leiden'] = leiden_clusters['Cluster']
        
        except FileNotFoundError:
            pass

        for name, labels in labels_list.items():
            print(f'Plotting {name} clustering...')
            label_color_dict = PlotLatentSpace(
                reconstructions=False,
                zoom=None,
                X_encoded_embedded=X_encoded_embedded,
                X_decoded_reversed=None,
                y=labels,  
                channel_color_dict=None,
                scatter_point_size=config.scatter_point_size,
                filename=name,
                save_dir=save_dir
            )

        ###########################################################################
        # decode latent vectors to visualize learned reconstructions

        X_decoded, X_decoded_reversed = DecodeVectors(
            decoder=decoder, X_encoded=X_encoded, X=X, X_seg=X_seg, sample_labels=sample_labels,
            bkgd_limits=bkgd_limits, contrast_limits=contrast_limits,
            channel_color_dict=config.channel_colors,
            tif_channels=config.tif_channels,
            patch_dims=patch_dims, mask=mask,
            chunk_size=chunk_size, intensity_multiplier=1.3
        )

        # plot latent vectors as their learned reconstructions
        PlotLatentSpace(
            reconstructions=True,
            zoom=2.5,
            X_encoded_embedded=X_encoded_embedded,
            X_decoded_reversed=X_decoded_reversed,
            y=None,
            channel_color_dict=config.channel_colors,
            scatter_point_size=config.scatter_point_size,
            filename='patches',
            save_dir=save_dir
        )

        # get input and output images of lassoed latent vectors for targeted analysis of vectors 
        if config.lasso_vector_tool:

            LassoVectors(
                contrast_limits=contrast_limits,
                patch_dims=patch_dims,
                imgs_instead_of_points=False,
                zoom=2.5,
                X=X,
                X_seg=X_seg,
                X_encoded=X_encoded,
                X_encoded_embedded=X_encoded_embedded,
                X_decoded_reversed=X_decoded_reversed,
                y=labels_list['sample'],
                numColumns=10,
                intensity_multiplier=1.1,
                tif_channels=config.tif_channels,
                channel_color_dict=config.channel_colors,
                max_examples=1000,
                patch_font_size=3.0,
                save_dir=save_dir
            )

        # plot learned reconstructions of input image patches
        PlotReconstructedImages(
            patch_dims=patch_dims,
            X=X,
            y=labels_list['sample'][0:100],
            X_seg=X_seg,
            X_decoded_reversed=X_decoded_reversed,
            contrast_limits=contrast_limits,
            numColumns=10,
            tif_channels=config.tif_channels,
            channel_color_dict=config.channel_colors,
            intensity_multiplier=1.1,
            patch_font_size=3.0,
            filename='learned_reconstructions',
            save_dir=save_dir
        )
           
        # compute mean squared error between input image patches and their learned reconstructions
        (average_error,
         errors,
         X_transform_outliers,
         y_outliers,
         X_outliers_seg,
         X_decoded_reversed_outliers,
         outlier_idxs,
         X_decoded_reversed_outliers) = mse(
            patch_dims=patch_dims,
            X_transform=X_transform,
            y=labels_list['sample'],
            X_seg=X_seg,
            X_decoded=X_decoded,
            X_decoded_reversed=X_decoded_reversed,
            mse_percentile_cutoff=99,
            filename='mse_dist',
            save_dir=save_dir
        )

        # visualize input image patches associated with poor learned reconstructions
        PlotReconstructedImages(
            patch_dims=patch_dims,
            X=X_transform_outliers,
            y=y_outliers,
            X_seg=X_outliers_seg,
            X_decoded_reversed=X_decoded_reversed_outliers,
            contrast_limits=contrast_limits,
            numColumns=10,
            tif_channels=config.tif_channels,
            channel_color_dict=config.channel_colors,
            intensity_multiplier=1.1,
            patch_font_size=3.0,
            filename='outliers',
            save_dir=save_dir
        )

        ###########################################################################
        # display interpolation grid of learned reconstructions
        
        if config.latent_dimension == 2:
            InterpolationGrid(
                patch_dims=patch_dims,
                grid_size=50,
                X_encoded=X_encoded,
                y=y[config.cluster_column],
                decoder=decoder,
                label_color_dict=label_color_dict,
                channel_color_dict=config.channel_colors,
                frac_of_scatter_points=1.0,
                scatter_point_size=config.scatter_point_size,
                make_sample_sizes_equal=False,
                img_brightness_multiplier=1.2,
                scatter_point_alpha=1.0,
                save_dir=save_dir
            )

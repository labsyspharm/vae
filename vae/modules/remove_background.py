import os
import yaml
import logging
import pathlib

import pandas as pd
import numpy as np

import zarr
import dask.array as da

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from ..gmm import get_gmm_and_pos_label
from ..utils import log_banner, log_multiline, log_transform


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


def flatten_and_log_transform(channel_patches, threshold, max_pixels):
    total_pixels = channel_patches.size
    
    # flatten image patches for raveling efficiency
    data = channel_patches.reshape(
        -1, np.prod(channel_patches.shape[1:])
    ) 
    if total_pixels > max_pixels:
        num_patches = int(max_pixels / data.shape[1])
        patch_selections = np.random.choice(
            data.shape[0], num_patches, replace=False
        )
        data = data[patch_selections]
        data = da.where(data < threshold, 0, data)
        data = log_transform(data)
        data = data.ravel().compute()
    else:
        data = data[:]
        data = da.where(data < threshold, 0, data)
        data = log_transform(data)
        data = data.ravel().compute()                       

    return data


def compute_anchors(raveled_data):
    """
    For computing 1D histogram fits
    Compute bunch of reference points
    min: min
    max: max
    mp0.01/mp0.05: mean of values <= 0.01/0.05 precentile
    mp99.99/mp99.95: mean of values >= 99.99/99.95 precentile
    gmm_peakX: mean of gaussian modes
    
    """
    out = {}
    non_zero_mask = np.where(raveled_data > 0)
    out['min'], out['max'] = (
        raveled_data[non_zero_mask].min(), 
        raveled_data[non_zero_mask].max()
    )
    out['mp0.01'], out['mp99.99'] = bottom_top(
        raveled_data[non_zero_mask], 0.01, 99.99
    )
    out['mp0.05'], out['mp99.95'] = bottom_top(
        raveled_data[non_zero_mask], 0.05, 99.95
    )
    out['gmm_peak1'], out['gmm_peak2'] = np.sort(
        get_gmm_and_pos_label(raveled_data)[0].means_.ravel()
    )

    return out


def bottom_top(raveled_data, p0, p1):
    """for computing 1D histogram fits"""
    
    p0, p1 = np.percentile(raveled_data, [p0, p1])
    if p0 == 0:
        vmin = raveled_data.min()
    else:
        vmin = np.mean(raveled_data, where=raveled_data <= p0)
    if p1 == 100:
        vmax = raveled_data.max()
    else:
        vmax = np.mean(raveled_data, where=raveled_data >= p1)
    return vmin, vmax


def save_figs(dpi=300, format='pdf', out_dir=None, prefix=None, close=True):
    figs = [plt.figure(i) for i in plt.get_fignums()]
    if prefix is not None:
        for f in figs:
            if f._suptitle:
                f.suptitle(f'{prefix} {f._suptitle.get_text()}')
            else:
                f.suptitle(prefix)
    names = [f._suptitle.get_text() if f._suptitle else "" for f in figs]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for f, n, nm in zip(figs, plt.get_fignums(), names):
        f.savefig(out_dir / f'{n}-{nm}.{format}', dpi=dpi, bbox_inches='tight')
        if close:
            plt.close(f)


max_pixels = 2_000_000


def REMOVE_BACKGROUND(config):
    
    if not os.path.exists(
      os.path.join(config.output_path, 
                   'checkpoints/REMOVE_BACKGROUND.txt')):

        print()
        Poly = np.polynomial.Polynomial
        np.random.seed(42)

        path = os.path.join(
            config.output_path,
            (f'3_cellcutter_output_win{config.window_size}/' +
             f'train_patches_{config.window_size}_qc.zip')
            )
        store = zarr.ZipStore(path, mode='r')
        z = da.from_zarr(zarr.open(store=store))

        # Read training labels
        cellcutter_input_path = os.path.join(
            config.output_path, '1_cellcutter_input'
        )
        csv_path = os.path.join(cellcutter_input_path, 'train_qc.csv')
        csv = pd.read_csv(csv_path)
        csv['Sample'] = csv['Sample'].astype(str)

        save_dir = os.path.join(config.output_path, '5_background_limits')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        #######################################################################
        # Generate dictionary to store channel limits
        
        if not os.path.exists(os.path.join(save_dir, 'bkgd_limits.yml')):
            
            bkgd_limits = {}
            for sample, _ in csv.groupby('Sample'):
                logger.info(f'Storing channel limits for sample {sample}')

                keys = [
                    i for i in zip([sample]*len(config.tif_channels), 
                                   config.tif_channels)
                ]

                values = [
                    i for i in [0]*len(config.tif_channels)
                ]

                for key, val in zip(keys, values):
                    bkgd_limits[key] = val

            f = open(os.path.join(save_dir, 'bkgd_limits.yml'), 'w')
            out = {
                str(k): v for k, v in bkgd_limits.items()
            }  # Avoiding yaml formatting errors
            yaml.dump(out, f, sort_keys=False, allow_unicode=False)

            print()
        else:
            bkgd_limits = yaml.safe_load(
                open(os.path.join(save_dir, 'bkgd_limits.yml'))
            )
            bkgd_limits = {eval(k): v for k, v in bkgd_limits.items()}
        
        #######################################################################
        # Vizualize histogram alignment
        
        for type in ['processed', 'raw']:
            
            logger.info(f'Generating {type} channel histograms plots...')
            
            num_channels = len(config.tif_channels)
            num_rows = num_columns = int(np.ceil(np.sqrt(num_channels)))

            fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 15))
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
            axs = axs.flatten()
            
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            
            for ch, marker in enumerate(config.tif_channels):
                if marker in config.tif_channels:  # Allow for debugging
                    ax = axs[ch]
                    handles = []
                    if marker in config.tif_channels:
                        logger.info(f'Plotting channel {marker}')
                        
                        for (sample, group), color in zip(
                          csv.groupby('Sample'), colors):

                            imgs = z[ch, group.index]
                            raveled_data = flatten_and_log_transform(
                                channel_patches=imgs, 
                                threshold=bkgd_limits[(sample, marker)],
                                max_pixels=max_pixels
                            )

                            if type == 'raw':
                                pass
                            
                            elif type == 'processed':
                                if str(imgs.dtype) == 'uint8':
                                    divisor = 255
                                elif str(imgs.dtype) == 'uint16':
                                    divisor = 65535

                                min_val = bkgd_limits[(sample, marker)]
                                log_min = np.log10(min_val + 1)
                                log_max = np.log10(divisor + 1)

                                range_vals = log_max - log_min
                                mask = raveled_data > log_min
                                scaled_values = (
                                    (raveled_data - log_min) / range_vals
                                )
                                raveled_data = da.where(mask, scaled_values, 0)
                                raveled_data = raveled_data[raveled_data > 0]
                                raveled_data = np.clip(raveled_data, 0, 1)
                            
                            if str(imgs.dtype) == 'uint8':
                                bins = 15
                            elif str(imgs.dtype) == 'uint16':
                                bins = 200 
                            
                            # Get percentile values
                            p_min, p_max = np.percentile(
                                raveled_data, [0.01, 99.99]
                            )
                            
                            # Ensure range >0 to avoid ValueError (np.linspace)
                            if p_min == p_max:
                                # If identical, create small range around value
                                p_min = max(0, p_min - 0.01)
                                p_max = p_max + 0.01
                                
                            bins = np.linspace(p_min, p_max, bins)
                            counts, bin_edges = np.histogram(
                                raveled_data, bins=bins
                            )
                            ax.step(
                                bin_edges[:-1], counts, where='mid', 
                                label=sample.split('-')[-1]
                            )
                            handle = mlines.Line2D(
                                [], [], color=color, marker='s', markersize=8, 
                                linestyle='None', label=sample.split('-')[-1]
                            )
                            handles.append(handle)
                            fig.suptitle(f'log10({type})', fontsize=12)
                            ax.set_title(marker, fontsize=10)
                            ax.tick_params(
                                axis='both', which='major', labelsize=7
                            )
                        
                        ax.legend(fontsize=8)
                        ax.legend(
                            handles=handles,
                            fontsize=8,
                            markerscale=1,
                            loc='best',
                            frameon=False,
                            handletextpad=0.1
                        )
            
            # Hide any unused subplots
            for ax in axs[num_channels:]:
                ax.set_visible(False)
            
            plt.tight_layout()
            print()
        
        save_figs(format='pdf', out_dir=os.path.join(save_dir, 'plots'))

import os
import pickle
import logging
import pathlib

import numpy as np

from natsort import natsorted

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


def ravel_data(channel_patches, threshold, max_pixels):
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
    compute bunch of reference points
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

# sample names here must match those specified in 1_cellcutter_input CSV files
thresholds = {
    ('CRC097', 'anti_CD3'): 2506,
    ('CRC102', 'anti_CD3'): 3173,
    ('C9', 'anti_CD3'): 1656,

    ('CRC097', 'anti_CD45RO'): 923,
    ('CRC102', 'anti_CD45RO'): 1846,
    ('C9', 'anti_CD45RO'): 1656,
    
    ('CRC097', 'Keratin_570'): 1385,
    ('CRC102', 'Keratin_570'): 1518,
    ('C9', 'Keratin_570'): 900,

    ('CRC097', 'aSMA_660'): 690,
    ('CRC102', 'aSMA_660'): 690,
    ('C9', 'aSMA_660'): 1104,

    ('CRC097', 'CD4_488'): 3863,
    ('CRC102', 'CD4_488'): 5105,
    ('C9', 'CD4_488'): 5077,
    
    ('CRC097', 'CD45_PE'): 2070,
    ('CRC102', 'CD45_PE'): 3266,
    ('C9', 'CD45_PE'): 828,
    
    ('CRC097', 'PD1_647'): 1932,
    ('CRC102', 'PD1_647'): 2345,
    ('C9', 'PD1_647'): 2647,

    ('CRC097', 'CD20_488'): 3863,
    ('CRC102', 'CD20_488'): 5381,
    ('C9', 'CD20_488'): 1380,
    
    ('CRC097', 'CD68_555'): 828,
    ('CRC102', 'CD68_555'): 851,
    ('C9', 'CD68_555'): 766,

    ('CRC097', 'CD8a_660'): 3193,
    ('CRC102', 'CD8a_660'): 7320,
    ('C9', 'CD8a_660'): 2655,
    
    ('CRC097', 'CD163_488'): 2621,
    ('CRC102', 'CD163_488'): 3144,
    ('C9', 'CD163_488'): 1484,

    ('CRC097', 'FOXP3_570'): 928,
    ('CRC102', 'FOXP3_570'): 1036,
    ('C9', 'FOXP3_570'): 515,
    
    ('CRC097', 'PDL1_647'): 2281,
    ('CRC102', 'PDL1_647'): 2841,
    ('C9', 'PDL1_647'): 1308,

    ('CRC097', 'Ecad_488'): 2705,
    ('CRC102', 'Ecad_488'): 2779,
    ('C9', 'Ecad_488'): 1651,
    
    ('CRC097', 'Vimentin_555'): 1055,
    ('CRC102', 'Vimentin_555'): 1080,
    ('C9', 'Vimentin_555'): 398,

    ('CRC097', 'CDX2_647'): 4154,
    ('CRC102', 'CDX2_647'): 3692,
    ('C9', 'CDX2_647'): 3940,
    
    ('CRC097', 'LaminABC_488'): 1550,
    ('CRC102', 'LaminABC_488'): 1846,
    ('C9', 'LaminABC_488'): 1363,

    ('CRC097', 'Desmin_555'): 1267,
    ('CRC102', 'Desmin_555'): 1572,
    ('C9', 'Desmin_555'): 1781,
    
    ('CRC097', 'CD31_647'): 758,
    ('CRC102', 'CD31_647'): 1111,
    ('C9', 'CD31_647'): 721,

    ('CRC097', 'PCNA_488'): 2023,
    ('CRC102', 'PCNA_488'): 2476,
    ('C9', 'PCNA_488'): 5846,
    
    ('CRC097', 'CollagenIV_647'): 1963,
    ('CRC102', 'CollagenIV_647'): 1367,
    ('C9', 'CollagenIV_647'): 8685,
}


def ALIGN_HISTOGRAMS(config):

    print()
    Poly = np.polynomial.Polynomial
    np.random.seed(42)
    
    # sample names here must match those specified in 1_cellcutter_input CSV files
    # reference image must come first
    zarrs = {
        'CRC097': '/n/scratch/users/g/gjb15/VAE9_VIG7_multi-tissue/test/'
        'CRC097/2_cellcutter_output_win14/train_thumbnails_14.zip', 
        'CRC102': '/n/scratch/users/g/gjb15/VAE9_VIG7_multi-tissue/test/'
        'CRC102/2_cellcutter_output_win14/train_thumbnails_14.zip',
        'C9': '/n/scratch/users/g/gjb15/VAE9_VIG7_multi-tissue/test/'
        'C9/2_cellcutter_output_win14/train_thumbnails_14.zip'
    }
    
    save_dir = os.path.join(config.output_path, '4_histogram_alignment')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if not os.path.exists(
      os.path.join(config.output_path, 'checkpoints/ALIGN_HISTOGRAMS.txt')):

        #######################################################################
        # generate dictionary of image channels and store channel limits
        
        imgs = {}
        limits = {}
        for e, (sample, path) in enumerate(zarrs.items()):
            store = zarr.ZipStore(path, mode='r')
            z = da.from_zarr(zarr.open(store=store))
            for channel, marker in enumerate(config.tif_channels):
                imgs[(sample, marker)] = z[channel]
            # if e == 0:  # assuming reference image is first in sample paths
            if not os.path.exists(
              os.path.join(save_dir, 'limits.pkl')):
                print(
                    f'Storing channel limits for sample {sample}'
                )
                keys = [
                    i for i in zip([sample]*len(config.tif_channels), 
                                   config.tif_channels)
                ]
                values = [
                    value for (key, value) in thresholds.items() if 
                    key[0] == sample
                ]

                for key, val in zip(keys, values):
                    limits[key] = val   

        if not os.path.exists(os.path.join(save_dir, 'limits.pkl')):
            with open(
                  os.path.join(
                   save_dir, 'limits.pkl'), 'wb') as handle:
                pickle.dump(
                        limits, handle, 
                        protocol=pickle.HIGHEST_PROTOCOL
                    ) 
            print()
        else:
            with open(os.path.join(
               save_dir, 'limits.pkl'), 'rb') as handle:
                limits = pickle.load(handle)
        
        #######################################################################
        # vizualize histogram alignment

        antibody_abbrs = {
            'anti_CD3': 'CD3', 'anti_CD45RO': 'CD45RO', 
            'Keratin_570': 'Keratin', 'aSMA_660': 'aSMA', 'CD4_488': 'CD4', 
            'CD45_PE': 'CD45', 'PD1_647': 'PD1', 'CD20_488': 'CD20', 
            'CD68_555': 'CD68', 'CD8a_660': 'CD8a', 'CD163_488': 'CD163', 
            'FOXP3_570': 'FOXP3', 'PDL1_647': 'PDL1', 'Ecad_488': 'ECAD', 
            'Vimentin_555': 'Vimentin', 'CDX2_647': 'CDX2', 
            'LaminABC_488': 'LaminABC', 'Desmin_555': 'Desmin', 
            'CD31_647': 'CD31', 'PCNA_488': 'PCNA', 
            'CollagenIV_647': 'CollagenIV'
        }
        
        for type in ['raw', 'processed']:
            print(f'Generating {type} channel histograms plot')
            fig, axs = plt.subplots(
                3, np.ceil(len(config.tif_channels) / 3).astype('int'),
                figsize=(15, 7)
            )
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            for marker, ax in zip(config.tif_channels, axs.ravel()):
                handles = []
                if marker in config.tif_channels:
                    print(f'Plotting channel {marker}')
                    for sample, color in zip(natsorted(zarrs.keys()), colors):
                        raveled_data = ravel_data(
                            channel_patches=imgs[(sample, marker)], 
                            threshold=thresholds[(sample, marker)],
                            max_pixels=max_pixels
                        )
                        if type == 'raw':
                            pass
                        else:
                            min_val = limits[(sample, marker)]
                            log_min = np.log10(min_val + 1)
                            log_max = np.log10(65535 + 1)

                            range_vals = log_max - log_min
                            mask = raveled_data > log_min
                            scaled_values = (
                                (raveled_data - log_min) / range_vals
                            )
                            raveled_data = da.where(mask, scaled_values, 0)
                            raveled_data = raveled_data[raveled_data > 0]
                            raveled_data = np.clip(raveled_data, 0, 1)
                        bins = np.linspace(
                            *np.percentile(
                                raveled_data, [0.01, 99.99]), 200
                        )
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
                        fig.suptitle(type, fontsize=12)
                        ax.set_title(antibody_abbrs[marker], fontsize=10)
                        ax.tick_params(axis='both', which='major', labelsize=7)
                    ax.legend(fontsize=8)
                    ax.legend(
                        handles=handles,
                        fontsize=8,
                        markerscale=1,
                        loc='best',
                        frameon=False,
                        handletextpad=0.1
                    )
            fig.tight_layout()
            print()
        
        save_figs(format='pdf', out_dir=os.path.join(save_dir, 'plots'))
        print()

        #######################################################################

import os
import sys
import glob
import yaml
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import dask.array as da
import zarr
from tensorflow import keras
import tensorflow.compat.v1 as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import load_model
from skimage.color import gray2rgb
from matplotlib.colors import to_rgb

# Local imports
from .deepexplain.tensorflow.methods import DeepExplain 
from .deepexplain.utils import preprocess
from ..utils import (
    log_banner, log_multiline, transposeZarr, compute_vignette_mask,
    log_transform, clip_outlier_pixels, 
    # reverse_processing
)

# Python's built-in random module is required 
# to obtain reproducible results
seed = 138
random.seed(seed)


def input_output(config):

    img_dims = (
        config.window_size,
        config.window_size, 
        len(config.tif_channels)
    )

    # Create save dir
    save_dir = os.path.join(
        config.output_path, f'8_saliency_map_LD{config.latent_dimension}'
    )
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Read combined training, validation, and test image patches
    combo_dir = os.path.join(
        config.output_path, 
        f'7_latent_space_LD{config.latent_dimension}/combined_zarr'
    )

    X = zarr.open(combo_dir)
    X = transposeZarr(z=X)
    
    # Load percentile cutoffs
    cutoffs_dir = os.path.join(
        config.output_path, '4_feature_preprocessing_selections'
    )
    try:
        with open(os.path.join(cutoffs_dir, 'cutoffs.pkl'), 'rb') as handle:
            percentile_cutoffs = pickle.load(handle)
    except FileNotFoundError:
        percentile_cutoffs = None
    
    # background_dir = os.path.join(
    #     config.output_path, '5_background_limits'
    # )
    # bkgd_limits = yaml.safe_load(
    #     open(os.path.join(background_dir, 'bkgd_limits.yml'))
    # )
    # bkgd_limits = {eval(k): v for k, v in bkgd_limits.items()}

    # Image contrast limits
    contrast_limits = yaml.safe_load(open(config.contrast_path))['setContrast']

    # Load previously saved encoder and decoder
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

    # Read Leiden clusters if they have been computed previously
    leiden_path = os.path.join(
        config.output_path,
        f'7_latent_space_LD{config.latent_dimension}/'
        'encodings_FULL-patches.csv')
    try:
        clusters = pd.read_csv(leiden_path)
    except FileNotFoundError:
        # Use HDBSCAN clusters to compute concept saliency
        clusters = np.load(
            os.path.join(
                config.output_path, 
                f'7_latent_space_LD{config.latent_dimension}/hdbscan_labels.npy')
        )
        clusters = pd.DataFrame(data={'Cluster': clusters})
    print()

    # Read encodings
    if not os.path.exists(os.path.join(save_dir, 'concept_vectors.pkl')) or \
       not os.path.exists(os.path.join(save_dir, 'concept_scores.pkl')):
      
        print()
        print('Reading encodings...')

        encoding_dir = os.path.join(
            config.output_path,
            f'7_latent_space_LD{config.latent_dimension}'
        )

        # Use glob to match the wildcard encodings file pattern
        parquet_files = glob.glob(os.path.join(
            encoding_dir,
            'encodings*.parquet')
        )
        # Check if any files were found
        if parquet_files:
            encodings = pd.read_parquet(parquet_files[0])  # use the first match
            
            # Add clusters to encodings dataframe
            encodings['cluster'] = clusters['Cluster']
        else:
            raise FileNotFoundError("No encoding parquet file found.")
            sys.exit(1)
        
        print()
    else:
        encodings = None
    
    return (img_dims, save_dir, X, percentile_cutoffs, 
            contrast_limits, encodings, encoder, decoder, clusters)


# ARCHIVAL REVERSE PROCESSING FUNCTION
def reverse_processing(percentile_cutoffs, channel_slice, channel_name, contrast_limits):
    """Reverses percentile normalization and log10-transformation,
       pixel outliers remained clipped)."""

    if percentile_cutoffs is not None:
        lower_cutoff_log, upper_cutoff_log = percentile_cutoffs[channel_name]

        # Reverse percentile normalization
        channel_slice = (
            (((upper_cutoff_log - lower_cutoff_log) * (channel_slice - 0)) /
             (1 - 0)) + lower_cutoff_log
        )

    # Reverse log10-transform
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


def DecodeVectors(decoder, img_dims, concept_vectors, percentile_cutoffs, contrast_limits, tif_channels, channel_color_dict, intensity_multiplier):

    decoded_concept_vectors = {}

    for concept, cv in concept_vectors.items():
        
        bar = {}

        decoded = decoder.predict(cv)

        reconstructed_img = decoded.reshape(
            img_dims[0], img_dims[1], img_dims[2])

        # Initialize image overlay
        overlay = np.zeros(
            (reconstructed_img.shape[0], reconstructed_img.shape[1])
        )

        overlay = gray2rgb(overlay)

        for name, color in channel_color_dict.items():

            ch = tif_channels.index(name)

            channel_slice = reconstructed_img[:, :, ch]

            channel_slice = reverse_processing(
                percentile_cutoffs, channel_slice, name, contrast_limits
            )

            bar[name] = channel_slice.sum()
            
            # 0-1 normalize channel intensities per concept
            # min_val = min(bar.values())
            # max_val = max(bar.values())
            # bar = {
            #     k: (v - min_val) / (max_val - min_val) for 
            #     k, v in bar.items()
            # }       

            channel_slice = gray2rgb(channel_slice)

            channel_slice = channel_slice * intensity_multiplier

            overlay += channel_slice * to_rgb(color)

        overlay = overlay.reshape(
            (img_dims[0], img_dims[1], 3)
        )

        decoded_concept_vectors[concept] = (overlay, bar)
    
    # 0-1 normalize bar chart intensities across concepts
    all_bar_vals = [
        value for _, bar in decoded_concept_vectors.values() for 
        value in bar.values()
    ]
    min_bar_val = min(all_bar_vals)
    max_bar_val = max(all_bar_vals)
    for concept, (overlay, bar) in decoded_concept_vectors.items():
        for key in bar.keys():
            bar[key] = (bar[key] - min_bar_val) / (max_bar_val - min_bar_val)

    return decoded_concept_vectors


def SALIENCY_MAP(config):
 
    sess = K.get_session()
    with DeepExplain(session=sess, graph=sess.graph) as de:
        
        print()
        
        (img_dims, save_dir, X, percentile_cutoffs, contrast_limits,
         encodings, encoder, decoder, clusters) = input_output(config)

        # Concepts to analyze
        concepts = sorted(clusters['Cluster'].unique())
        
        # VAE9_VIG7
        X_intensity_multiplier = 1.5
        decoded_intensity_multiplier = 2.75
        rectgrad_percentile_threshold = 20
        occlusion_window = (3, 3, 1)
        q1, q2 = (0.0, 99.7)  # lower/upper attribution score thresholds

        # VAE30
        # X_intensity_multiplier = 1.0
        # decoded_intensity_multiplier = 1.25
        # rectgrad_percentile_threshold = 40
        # occlusion_window = (16, 16, 1)
        # q1, q2 = (0.1, 99.6)  # lower/upper attribution score thresholds
        
        # Quantile of positive concept distribution to sample image patches from
        scores_quantile = 0.95
        
        # Number of top scoring image patches to plot
        n_top_scores = 15
        
        # AVAILABLE ATTRIBUTION METHODS ARE:
        # 'rectgrad', RectifiedGradient
        # 'rectgradmod', RectifiedGradientMod
        # 'rectgradconst', RectifiedGradientConst
        # 'rectgradprr', RectifiedGradientPRR
        # 'saliency', Saliency
        # 'grad*input', GradientXInput
        # 'deconv', Deconvolution
        # 'guidedbp', GuidedBackpropagation
        # 'smoothgrad', SmoothGrad
        # 'intgrad', IntegratedGradients
        # 'elrp', EpsilonLRP
        # 'deeplift', DeepLIFTRescale  # haven't got this one to work yet
        # 'occlusion', Occlusion

        method = 'rectgrad'

        antibody_abbrs = {
            'anti_CD3': 'CD3', 'anti_CD45RO': 'CD45RO',
            'Keratin_570': 'Keratin', 'aSMA_660': 'aSMA',
            'CD4_488': 'CD4', 'CD45_PE': 'CD45', 'PD1_647': 'PD1', 
            'CD20_488': 'CD20', 'CD68_555': 'CD68', 'CD8a_660': 'CD8a',
            'CD163_488': 'CD163', 'FOXP3_570': 'FOXP3', 'PDL1_647': 'PDL1',
            'Ecad_488': 'ECAD', 'Vimentin_555': 'Vimentin', 'CDX2_647': 'CDX2',
            'LaminABC_488': 'LaminABC', 'Desmin_555': 'Desmin', 
            'CD31_647': 'CD31', 'PCNA_488': 'PCNA',
            'CollagenIV_647': 'CollagenIV'
        }

        ############################################################
        # Preprocess image patches
        if percentile_cutoffs is not None:
            X_transform = clip_outlier_pixels(
                log_transform(X), percentile_cutoffs=percentile_cutoffs
            )
        else:
            X_transform = log_transform(X)
        
        mask, vmin, vmax = compute_vignette_mask(
            window_size=config.window_size, std_dev=config.mask_std_dev
        )
        if config.masked_model:
            print('Applying Gaussian vignette mask.')
            print()
            X_transform *= mask

        if not os.path.exists(os.path.join(save_dir, 'concept_vectors.pkl')):

            # Select image patch encodings with highest average cosine
            # similarities to all other image patches in a given cluster
            concept_data = {}
            for concept in concepts:
                if concept != -1:

                    print(
                        'Computing cosine similarity scores for '
                        f'cluster {concept} encodings...')

                    idxs = clusters.index[clusters['Cluster'] == concept]
                    z_plus = encodings.loc[idxs].drop(
                        columns=['CellID', 'cluster']
                    )

                    mean_encoding = z_plus.mean()

                    cos = [
                        cosine_similarity(
                            mean_encoding.values.reshape(1, -1), i[1].values.reshape(1, -1)
                        )
                        for i in z_plus.iterrows()
                    ]
                    
                    top = pd.Series([i.item() for i in cos]).sort_values(ascending=False)

                    concept_data[concept] = z_plus.iloc[top[0:1].index]
            print()
            
            ############################################################
            # Compute concept vectors (Zc) 
            # (Step 4 of the concept-saliency-maps algorithm in Brocki, 2019.)
            
            concept_vectors = {}
            for concept, z_plus in concept_data.items():

                print(f'Computing cluster {concept} concept vector...')
                
                minus_list = []
                for concept2, z_plus2 in concept_data.items():
                    if concept2 != concept:
                        
                        minus_list.append(z_plus2)

                if len(minus_list) >= 1:
                    z_minus = pd.concat(minus_list, axis=0, ignore_index=True)
                    zc = np.array([z_plus.mean() - z_minus.mean()])
                else:
                    zc = np.array([z_plus.mean()]) 

                concept_vectors[concept] = zc

            with open(
                    os.path.join(save_dir, 'concept_vectors.pkl'), 'wb') as handle:
                pickle.dump(
                    concept_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print()

        else:
            print('Loading concept vectors...')
            print()
            with open(
                    os.path.join(save_dir, 'concept_vectors.pkl'), 'rb') as handle:
                concept_vectors = pickle.load(handle)
        
        ############################################################
        # Decode concept vectors and view their learned reconstructions
        
        # print('Decoding concept vectors...')

        # decoded_cvs = DecodeVectors(
        #     decoder=decoder,
        #     img_dims=img_dims,
        #     concept_vectors=concept_vectors,
        #     percentile_cutoffs=percentile_cutoffs, 
        #     contrast_limits=contrast_limits,
        #     tif_channels=config.tif_channels,
        #     channel_color_dict=config.channel_colors,
        #     intensity_multiplier=decoded_intensity_multiplier
        # )
        
        # fig = plt.figure(figsize=(11, 3.9))  # VAE9_VIG7
        # # fig = plt.figure(figsize=(9, 2.7))  # VAE30
        
        # rows_rounded_up = len(concepts) // 8 + bool(len(concepts) % 8)
        # height_ratios = [1] * rows_rounded_up
        
        # rows, cols = (rows_rounded_up, 8)
        # # rows, cols = (3, 8)  # VAE9_VIG7   
        # # rows, cols = (2, 6)  # VAE30
        
        # outer_gs = gridspec.GridSpec(
        #     rows, cols, height_ratios=height_ratios, hspace=0.1, wspace=0.7
        # )  # VAE9_VIG7
        
        # # outer_gs = gridspec.GridSpec(
        # #     rows, cols, height_ratios=[1, 1], hspace=0.1, wspace=0.7
        # # )  # VAE30

        # for concept in decoded_cvs.keys():

        #     overlay = np.zeros((img_dims[0], img_dims[1]))
        #     overlay = gray2rgb(overlay)

        #     inner_gs = gridspec.GridSpecFromSubplotSpec(
        #         2, 1, subplot_spec=outer_gs[concept], 
        #         height_ratios=[0.3, 1], hspace=-0.2
        #     )
            
        #     ax1 = fig.add_subplot(inner_gs[0])
        #     ax1.bar(
        #         x=range(len(config.channel_colors)),
        #         height=[val for val in decoded_cvs[concept][1].values()],
        #         color=[config.channel_colors[name] for name in
        #                decoded_cvs[concept][1].keys()],
        #         width=0.8
        #     )
        #     ax1.set_xticks([])
        #     ax1.set_ylim(0, 1)
        #     ax1.set_yticks([0, 0.5, 1])
        #     ax1.set_ylabel(
        #         'Intensity', fontsize=4.5,
        #         fontweight='normal', labelpad=1.0
        #     )
        #     ax1.tick_params(axis='y', labelsize=5)
        #     ax1.spines['top'].set_visible(False)
        #     ax1.spines['bottom'].set_visible(False)
        #     ax1.spines['right'].set_visible(False)
        #     ax1.spines['left'].set_linewidth(0.4)

        #     ax2 = fig.add_subplot(inner_gs[1])
        #     ax2.imshow(decoded_cvs[concept][0])
        #     ax2.text(
        #         0.5, -0.05, f'Concept {concept}', fontsize=7, fontweight='bold',
        #         ha='center', va='top', transform=ax2.transAxes
        #     )
        #     ax2.axis('off')

        # legend_elements = []
        # for name, color in config.channel_colors.items():
        #     legend_elements.append(
        #         Line2D([0], [0], color=color, lw=5, label=name)
        #     )
        # fig.legend(
        #     handles=legend_elements, prop={'size': 8},
        #     bbox_to_anchor=(0.99, 0.99), handlelength=1.0,
        #     frameon=False
        # )
        
        # # VAE9_VIG7
        # plt.subplots_adjust(
        #     left=0.04, right=0.87,
        #     bottom=0.02, top=0.97,
        #     hspace=0.0, wspace=0.0
        # ) 

        # # VAE30
        # # plt.subplots_adjust(
        # #     left=0.04, right=0.84, 
        # #     bottom=0.01, top=0.97,
        # #     hspace=0.0, wspace=0.0
        # # )  

        # plt.savefig(
        #     os.path.join(save_dir, 'decoded_concept_vectors.pdf')
        # )
        # plt.close('all')
        # print()

        ############################################################
        # Compute concept scores (Sc) for latent vectors
        # (Step 5 of the concept-saliency-maps algorithm in Brocki, 2019.)
        
        if not os.path.exists(os.path.join(save_dir, 'concept_scores.pkl')):
            
            concept_scores = {}
            
            for concept, cv in concept_vectors.items():

                print(f'Computing cluster {concept} concept scores...')

                idxs = clusters.index[clusters['Cluster'] == concept]
                z_plus = encodings.loc[idxs].drop(
                    columns=['CellID', 'cluster']
                )

                idxs = clusters.index[clusters['Cluster'] != concept]
                z_minus = encodings.loc[idxs].drop(
                    columns=['CellID', 'cluster']
                )

                scores_plus = [np.sum(cv * j) for j in np.array(z_plus)]
                scores_minus = [np.sum(cv * j) for j in np.array(z_minus)]
                
                concept_scores[concept] = {
                    'plus': scores_plus, 'minus': scores_minus, 
                    'plus_idxs': z_plus.index
                }
            
            with open(
                    os.path.join(save_dir, 'concept_scores.pkl'), 'wb') as handle:
                pickle.dump(
                    concept_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print()
        else:
            print('Loading concept scores...')
            print()
            with open(
                    os.path.join(save_dir, 'concept_scores.pkl'), 'rb') as handle:
                concept_scores = pickle.load(handle)
        
        ############################################################
        # Isolate row indices for image patches with top concept scores
        
        if not os.path.exists(os.path.join(save_dir, 'top_scores.pkl')):    
            
            top_scores = {}
            
            for concept in concept_scores.keys():

                print(
                    'Isolating indices for patches with top concept' 
                    f' scores for concept {concept}...')
                
                scores_pls_df = pd.DataFrame(
                    concept_scores[concept]['plus'],
                    index=concept_scores[concept]['plus_idxs'],
                    columns=['score']
                )
                quantile = scores_pls_df['score'].quantile(scores_quantile)
                scores_pls_df['abs_diff'] = abs(scores_pls_df['score'] - quantile)
                sorted_scores = scores_pls_df.sort_values('abs_diff')
                nearest_idxs = sorted_scores.head(n_top_scores).index
                top_scores[concept] = tuple(nearest_idxs)
            
            with open(
                    os.path.join(save_dir, 'top_scores.pkl'), 'wb') as handle:
                pickle.dump(
                    top_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print()
        else:
            print('Loading indices for patches with top concept scores...')
            print()
            with open(
                    os.path.join(save_dir, 'top_scores.pkl'), 'rb') as handle:
                top_scores = pickle.load(handle)
        
        ############################################################
        # Assess separation between concept score distributions
        # for image patches with and without a given concept label
        
        print('Plotting concept score histograms...')
        
        # VAE9_VIG7
        fig, axs = plt.subplots(3, 8, figsize=(11.5, 3.5))

        # VAE30 
        # fig, axs = plt.subplots(2, 6, figsize=(9, 2.5))
        
        axs = axs.flatten()
        
        minus_color = 'tab:blue'
        plus_color = 'tab:orange'

        concept_counter = 0
        for idx, ax in enumerate(axs):
            try:
                axs[idx].hist(
                    concept_scores[concept_counter]['minus'], 
                    bins=50, density=True, color=minus_color,
                    alpha=0.75, lw=0.0, label='Concept neg.'
                )
                axs[idx].hist(
                    concept_scores[concept_counter]['plus'],
                    bins=50, density=True, color=plus_color, 
                    alpha=0.75, lw=0.0, label='Concept pos.'
                )
                axs[idx].set_title(
                    rf'$\bf{{Concept\ {concept_counter}}}$ scores',
                    fontsize=7, pad=4.0
                )
                axs[idx].set_ylabel('Density', fontsize=7)
                axs[idx].tick_params(axis='both', labelsize=5)
            except KeyError:
                ax.remove()
            concept_counter += 1

        legend_elements = [
            Line2D([0], [0], color=minus_color, 
                   alpha=0.75, lw=6, label='Concept neg.'),
            Line2D([0], [0], color=plus_color, alpha=0.75, 
                   lw=6, label='Concept Pos.'),
        ]

        # VAE9_VIG7
        fig.legend(
            handles=legend_elements, prop={'size': 5}, 
            bbox_to_anchor=(0.115, 0.95), handlelength=1, 
            frameon=False
        )
        plt.subplots_adjust(
            left=0.05, right=0.99,
            bottom=0.05, top=0.95, 
            hspace=0.4, wspace=0.6
        )  

        # VAE30
        # fig.legend(
        #     handles=legend_elements, prop={'size': 5}, 
        #     bbox_to_anchor=(0.143, 0.935), handlelength=1, 
        #     frameon=False
        # )
        # plt.subplots_adjust(
        #     left=0.06, right=0.99, 
        #     bottom=0.07, top=0.93, 
        #     hspace=0.4, wspace=0.6
        # )

        plt.savefig(os.path.join(save_dir, 'concept_score_hists.pdf'))
        plt.close('all')
        print()

        ############################################################
        # Create saliency maps with respect to concept vectors
        # for image patches with top concept scores
        
        input_layer = keras.Input(shape=(img_dims))
        latent = encoder(input_layer)
        
        concept_scores = [
            K.sum(cv * latent) for (concept, cv) in 
            concept_vectors.items()
        ]

        # THE KEY TO GETTING DEEPEXPLAIN LEGACY CODE TO WORK WITH TF2
        # IS USING "tensorflow.compat.v1" AND THIS NEXT LINE!
        tf.initialize_all_variables().run(session=sess)  

        for e, (concept, img_idxs) in enumerate(top_scores.items()):
            if concept in concepts:
                for img_idx in img_idxs:
                    
                    img = da.expand_dims(X_transform[img_idx], axis=0).compute()

                    # For percentile gradient methods
                    if method in ['rectgrad', 'rectgradmod', 'rectgradprr']: 
 
                        attrs = de.explain(
                            method, concept_scores[e], input_layer, img,
                            percentile=rectgrad_percentile_threshold
                        )
                    
                    elif method == 'occlusion':  # for perturbation methods
                        attrs = de.explain(
                            method, concept_scores[e], input_layer, img,
                            window_shape=occlusion_window, step=(1, 1, 1)
                        )
                    
                    else:
                        attrs = de.explain(
                            method, concept_scores[e], input_layer, img,
                        )

                    (attrs_processed, 
                     attrs_threshold_low,
                     attrs_threshold_high) = preprocess(attrs, q1, q2)

                    if config.masked_model:  # undo vignette mask

                        # Mask function is designed to be applied to all 
                        # cells in a batch during model training. Slicing first 
                        # dimension in this case to apply to a single patch.
                        img /= mask[0, :, :, :]
                        # pass
                    
                    # Visualize attributions
                    fig = plt.figure(figsize=(13, 4.5))
                    
                    rows, cols = (4, 6)
                    outer_gs = gridspec.GridSpec(
                        rows, cols, height_ratios=[1, 1, 1, 1],
                        hspace=0.0, wspace=0.1
                    )

                    merge_channels = []
                    for ch, channel_name in enumerate(config.tif_channels):

                        inner_gs = gridspec.GridSpecFromSubplotSpec(
                            1, 2, subplot_spec=outer_gs[ch], 
                            hspace=0.0, wspace=0.05
                        )
                        
                        overlay = np.zeros((img_dims[0], img_dims[1]))
                        overlay = gray2rgb(overlay)

                        channel_slice = img[0, :, :, ch]

                        if channel_slice.max() > 0.3:
                            merge_channels.append(channel_name)
                        
                        channel_slice = reverse_processing(
                            percentile_cutoffs, channel_slice, 
                            channel_name, contrast_limits
                        )

                        channel_slice = gray2rgb(channel_slice)
                        
                        channel_slice = channel_slice * X_intensity_multiplier
                        
                        overlay += channel_slice * to_rgb([1, 1, 1])
                                
                        # Fluorescence image
                        ax1 = fig.add_subplot(inner_gs[0])

                        overlay = np.clip(overlay, 0, 1)
                        
                        ax1.imshow(overlay)
                        ax1.set_title(
                            f'{antibody_abbrs[channel_name]}',
                            fontsize=7, fontweight='bold', y=0.94
                        )
                        ax1.axis('off')

                        # Attributions image
                        norm = TwoSlopeNorm(
                            vcenter=0, vmin=attrs_threshold_low, 
                            vmax=attrs_threshold_high
                        )
                        ax2 = fig.add_subplot(inner_gs[1])
                        attr_img = ax2.imshow(
                            attrs_processed[0, :, :, ch], 
                            cmap='coolwarm', norm=norm
                        ) 
                        ax2.set_title(method, fontsize=7, y=0.94)
                        ax2.axis('off')

                        if ch == len(config.tif_channels) - 1:  # on last channel
                            
                            # Add cbar
                            pos = ax2.get_position()

                            cbar_left = pos.x1 - 0.04
                            cbar_bottom = pos.y0 - 0.09
                            cbar_width = 0.01
                            cbar_height = pos.height + 0.03

                            cbar_ax = fig.add_axes(
                                [cbar_left, cbar_bottom, cbar_width, cbar_height]
                            )
                            cbar = fig.colorbar(attr_img, cax=cbar_ax)
                            cbar.set_label('attribution score', fontsize=6)
                            cbar.ax.tick_params(labelsize=6)

                            last_gs = gridspec.GridSpecFromSubplotSpec(
                                1, 2, subplot_spec=outer_gs[ch + 1], 
                                hspace=0.0, wspace=0.0
                            )

                            merge = np.zeros((img_dims[0], img_dims[1]))
                            merge = gray2rgb(merge)
                            
                            for ch2, channel_name2 in enumerate(config.tif_channels):
                                if channel_name2 in merge_channels:
                                    
                                    channel_slice = img[0, :, :, ch2]
                                    
                                    channel_slice = reverse_processing(
                                        percentile_cutoffs, channel_slice, 
                                        channel_name2, contrast_limits
                                    )

                                    channel_slice = gray2rgb(channel_slice)
                                    
                                    channel_slice = (
                                        channel_slice * X_intensity_multiplier
                                    )

                                    try:
                                        merge += channel_slice * to_rgb(
                                            config.channel_colors[channel_name2]
                                        )
                                    except KeyError:
                                        pass

                            ax3 = fig.add_subplot(last_gs[1])

                            merge = np.clip(merge, 0, 1)
                            
                            ax3.imshow(merge)
                            ax3.set_title(
                                'Merge', fontsize=7, 
                                fontweight='bold', y=0.94
                            )
                            ax3.axis('off')

                    legend_elements = []
                    for name, color in config.channel_colors.items():
                        legend_elements.append(
                            Line2D([0], [0], color=color, lw=5, 
                                   label=antibody_abbrs[name])
                        )
                    fig.legend(
                        handles=legend_elements, prop={'size': 8},
                        bbox_to_anchor=(1.0, 0.99), handlelength=1.0,
                        frameon=False
                    )
                    
                    plt.subplots_adjust(
                        left=0.01, right=0.93,
                        bottom=0.01, top=0.99,
                        hspace=0.0, wspace=0.0
                    )

                    plt.savefig(
                        os.path.join(
                            save_dir, f'concept{concept:02}_cellID_{img_idx}.pdf'
                        )
                    )
                    plt.close('all')

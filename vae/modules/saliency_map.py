import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import zarr

import tensorflow as tf
from keras.models import load_model
import tensorflow.python.keras.backend as K

from .deepexplain.tensorflow.methods import DeepExplain
from .deepexplain.utils import preprocess

from ..utils import log_banner, log_multiline, transposeZarr

K.clear_session()
np.random.seed(237)
sess = K.get_session()


def SALIENCY_MAP(config):
    save_dir = os.path.join(config.output_path, f'7_saliency_map_LD{config.latent_dimension}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # read encodings 
    encodings = pd.read_csv(
        os.path.join(config.output_path, 
                     f'6_latent_space_LD{config.latent_dimension}/'
                     f'{config.output_path.name}_encodings.csv')
    )
    
    # read leiden clusters
    clusters = pd.read_csv(
        os.path.join(config.output_path, 
                     f'6_latent_space_LD{config.latent_dimension}/'
                     f'{config.output_path.name}_encodings-patches_res1.5.csv')
    )

    # add clusters to encodings dataframe
    encodings['cluster'] = clusters['Cluster']

    # read combined training, validation, and test image patches
    combo_dir = os.path.join(
        config.output_path, f'6_latent_space_LD{config.latent_dimension}/combined_zarr'
    )
    X = zarr.open(combo_dir)
    X = transposeZarr(z=X)

    # compute the concept vector (Zc) for each cluster 
    # (step 4 of the concept-saliency-maps algorithm in Brocki, 2019)
    concept_vectors = {}
    for clus in sorted(encodings['cluster'].unique()):
    
        z_plus = encodings[encodings['cluster'] == clus].drop(columns=['CellID', 'cluster']) 
        z_minus = encodings[encodings['cluster'] != clus].drop(columns=['CellID', 'cluster'])

        zc = z_plus.mean() - z_minus.mean()
        
        concept_vectors[clus] = zc

    # calculate the concept score (Sc) for each latent vector
    # (step 4 of the concept-saliency-maps algorithm in Brocki, 2019)
    concept_scores = pd.DataFrame()
    for clus in sorted(encodings['cluster'].unique()):
        print(f'computing concept scores for cluster {clus}...')
        cv = concept_vectors[clus]
        cs = np.dot(encodings.drop(columns=['CellID', 'cluster'], inplace=False), cv)
        concept_scores[f'concept_{clus}'] = cs

    concept_scores['cluster'] = encodings['cluster']

    # identify the maximum score for each cluster of image patches
    mydict = {}
    for clus, group in sorted(concept_scores.groupby('cluster')):

        group = group.drop(columns=['cluster'])
        max_index = np.argmax(group.mean())
        mydict[clus] = max_index

    # generate saliency maps
    with DeepExplain(session=sess) as de:
        
        # load previously saved encoder and decoders
        try:
            encoder = load_model(os.path.join(config.output_path, '5_train_vae/encoder.hdf5'))
        except OSError:
            print('Encoder not found.')
            sys.exit()

        try:
            decoder = load_model(os.path.join(config.output_path, '5_train_vae/decoder.hdf5'))
        except OSError:
            print('Decoder not found.')
            sys.exit()

        k = 95
        method = 'rectgrad'
        input_layer = encoder.input  # place holder since we need to pick our layer from encoder
        # input_layer = tf.compat.v1.placeholder(tf.float32, [10, 14, 14, 21])
      
        imgs = X[0:10]
        scores = concept_scores.drop(columns=['cluster'])[0:10]

        attribution = [de.explain(method, score, input_layer, img, percentile=k) for score, img in zip(scores, imgs)]
        print(attribution)

## view saliency maps
# f, axs = plt.subplots(3,3, figsize=(15,15))

# axs[0, 0].matshow(decoded[0])
# axs[0, 1].matshow(decoded[1])
# axs[0, 2].matshow(decoded[2])

# gene1 = 1
# gene2 = 1
# gene3 = 1

# axs[1, 0].set_title(top_correlation[0].index[gene1], fontsize=20)
# axs[1, 0].imshow(imgs[0][gene1,:,:,0])
# axs[2, 0].imshow(preprocess(attribution[0],0.5,99.5)[gene1])

# axs[1, 1].set_title(top_correlation[1].index[gene2], fontsize=20)
# axs[1, 1].imshow(imgs[1][gene2,:,:,0])
# axs[2, 1].imshow(preprocess(attribution[1],0.5,99.5)[gene2])

# axs[1, 2].set_title(top_correlation[2].index[gene3], fontsize=20)
# axs[1, 2].imshow(imgs[2][gene3,:,:,0])
# axs[2, 2].imshow(preprocess(attribution[2],0.5,99.5)[gene3])

# plt.setp(axs, xticks=[], yticks=[])

# plt.subplots_adjust(hspace=0.05, wspace=0.2)
# plt.savefig('heatmaps_top.png')
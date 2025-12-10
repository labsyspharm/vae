import os
import logging

import pandas as pd

from ..utils import log_banner, log_multiline

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


def GENERATE_CELLCUTTER_INPUT(config):
    
    if not os.path.isfile(
       os.path.join(config.output_path,
                    'checkpoints/GENERATE_CELLCUTTER_INPUT.txt')):
        
        save_dir = os.path.join(config.output_path, '1_cellcutter_input')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        extension = os.path.splitext(config.csv_path)[1]

        if extension == '.parquet':
            csv = pd.read_parquet(config.csv_path)
        elif extension == '.csv':
            csv = pd.read_csv(config.csv_path)
        else:
            raise ValueError(f'Note: extension type {extension} not supported.')

        #######################################################################
        # weighted random sampling
        
        # drop noisy cells from HDBSCAN clustering
        csv = csv[csv['cluster_3d'] != -1]
        
        # calculate weighted random sample by cluster size (class balance)
        groups = csv.groupby('cluster_3d')
        sample_weights = pd.DataFrame(
            {'weights': 1 / (groups.size() * len(groups))}
        )
        weights = pd.merge(
            csv[['cluster_3d']], sample_weights,
            left_on='cluster_3d', right_index=True
        )

        csv = csv.sample(
            frac=config.percent_cells, replace=False,
            weights=weights['weights'], random_state=0, axis=0
        )

        # print()
        # print('Cells per cluster after cluster-weighted random sampling:')
        # print(csv.groupby('cluster_3d').size().sort_values(ascending=False))

        #######################################################################
        
        print()
        logger.info(
            'Partitioning CyLinter dataframe into training, ' 
            'validation, and test sets...'
        )
        print()
        
        # Shuffle csv data
        csv = csv.sample(frac=config.percent_cells, random_state=0)

        # Reserve 10% of data for testing after model training 
        split = round(len(csv) * 0.10)
        test = csv[0:split].copy()
        test.sort_values(by=['Sample', 'CellID'], inplace=True)
        
        # Reserve 10% of data for validation at the end of each training epoch
        validate = csv[split:split * 2].copy()
        validate.sort_values(by=['Sample', 'CellID'], inplace=True)
        
        # Use remaining data for model training
        train = csv[split * 2:].copy()
        train.sort_values(by=['Sample', 'CellID'], inplace=True)
    
        # Reset row indexes of each dataframe
        test.reset_index(drop=True, inplace=True)
        validate.reset_index(drop=True, inplace=True)
        train.reset_index(drop=True, inplace=True)

        #######################################################################

        # Save testing, validation, and training dataframes for cellcutter
        test.to_csv(os.path.join(save_dir, 'test_raw.csv'), index=False)
        validate.to_csv(os.path.join(save_dir, 'validate_raw.csv'), index=False)
        train.to_csv(os.path.join(save_dir, 'train_raw.csv'), index=False)

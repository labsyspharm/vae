import os
import logging

import pandas as pd

from subprocess import run

from ..utils import log_banner, log_multiline

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


def RUN_CELLCUTTER(config):

    cellcutter_input_path = os.path.join(
        config.output_path, '1_cellcutter_input'
    )

    save_dir = os.path.join(
        config.output_path, f'2_cellcutter_output_win{config.window_size}'
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    markers = pd.read_csv(config.markers_path)

    marker_channel_numbers = []
    for i in config.tif_channels:
        id = markers['channel_number'][markers['marker_name'] == i].values[0]
        marker_channel_numbers.append(str(id))

    for name in ['test', 'train', 'validate']:  
        
        if not os.path.exists(
            os.path.join(save_dir, 
                         f"{name}_thumbnails_{config.window_size}.zip")
        ):
            print()
            print(f'Cutting {name} data...')
            
            run(
                ["cut_cells", "-z", "-f", 
                 "--window-size", str(config.window_size),
                 "--cells-per-chunk", str(config.cells_per_chunk),
                 "--cache-size", str(config.cache_size_cellcutter), 
                 str(config.tif_path), str(config.mask_path),
                 os.path.join(cellcutter_input_path, f"{name}.csv"),
                 os.path.join(save_dir, 
                              f"{name}_thumbnails_{config.window_size}.zip"),
                 "--channels",  
                 ] + marker_channel_numbers
            )
        
        if not os.path.exists(
            os.path.join(save_dir, 
                         f"{name}_thumbnails_{config.window_size}_seg.zip")
        ):
            run(
                ["cut_cells", "-z", "-f", 
                 "--window-size", str(config.window_size),
                 "--cells-per-chunk", str(config.cells_per_chunk),
                 "--cache-size", str(config.cache_size_cellcutter),
                 str(config.outlines_path), str(config.mask_path),
                 os.path.join(cellcutter_input_path, f"{name}.csv"),
                 os.path.join(
                    save_dir, 
                    f"{name}_thumbnails_{config.window_size}_seg.zip"
                 ),
                 "--channels", "1"
                 ]
            )

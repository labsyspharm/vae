import logging

import os
import pandas as pd

from subprocess import run

from ..utils import log_banner, log_multiline

logger = logging.getLogger(__name__)
# log_multiline(logger.info, pd.DataFrame().to_string(index=False))


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

        cellcutter_marker_ids = []
        for i in config.tif_channels:
            id = markers['channel_number'][
                markers['marker_name'] == i].values[0]
            cellcutter_marker_ids.append(str(id))

        for name in ['test', 'train', 'validate']:
            print()
            print(f'Cutting {name} data...')
            run(
                ["cut_cells", "-z", "--window-size", str(config.window_size),
                 "--cells-per-chunk", str(config.cells_per_chunk),
                 "--cache-size", str(config.cache_size_cellcutter), str(config.tif_path),
                 str(config.mask_path),
                 os.path.join(cellcutter_input_path, f"{name}.csv"),
                 os.path.join(
                    save_dir, f"{name}_thumbnails_{config.window_size}.zarr"),
                 "--channels"] + cellcutter_marker_ids
                )

            run(
                ["cut_cells", "-z", "--window-size", str(config.window_size),
                 "--cells-per-chunk", str(config.cells_per_chunk),
                 "--cache-size", str(config.cache_size_cellcutter),
                 str(config.outlines_path), str(config.mask_path),
                 os.path.join(cellcutter_input_path, f"{name}.csv"),
                 os.path.join(
                    save_dir,
                    f"{name}_thumbnails_{config.window_size}_seg.zarr"),
                 "--channels", "1"
                 ]
                )

import os
import glob
import logging
import tempfile

import zarr
import tifffile
import ome_types
import numpy as np
import pandas as pd
from subprocess import run

from ..utils import log_banner, log_multiline

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


def append_artifact_mask_to_tif(img_path, mask_path):
    """Create a temporary file to store TIFF with artifact
    mask appended as the last channel.
    """
    
    temp_tif = tempfile.NamedTemporaryFile(
        suffix='.tif', delete=False
    )
    temp_tif_path = temp_tif.name
    temp_tif.close()

    # Open the original TIFF and read metadata only
    with tifffile.TiffFile(img_path) as tif:
        _, img_height, img_width = tif.series[0].shape
        
        mask = tifffile.imread(mask_path)
        mask = mask[np.newaxis, ...]  
        mask_trim = mask[:, 0:img_height, 0:img_width]

        channels = [
            page.asarray(out='memmap') for page in tif.pages
        ]
        stacked = np.stack(channels, axis=0)  # shape:(C,H,W)
        combined = np.concatenate([stacked, mask_trim], axis=0)

        with tifffile.TiffWriter(
             temp_tif_path, bigtiff=True) as tif_writer:
            tif_writer.write(
                combined, metadata={'axes': 'CYX'}
            )

    return temp_tif_path


def RUN_CELLCUTTER(config):
    
    cellcutter_input_path = os.path.join(
        config.output_path, '1_cellcutter_input'
    )

    artifact_detection_path = os.path.join(
        config.output_path, '2_artifact_detection'
    )

    save_dir = os.path.join(
        config.output_path, f'3_cellcutter_output_win{config.window_size}'
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    markers = pd.read_csv(config.markers_path)

    # Get the channel numbers for markers in config.tif_channels
    marker_channel_numbers = []
    for i in config.tif_channels:
        idx = markers['channel_number'][markers['marker_name'] == i].values[0]
        marker_channel_numbers.append(str(idx))
    
    # Add checkpoint tracking file
    checkpoint_file = os.path.join(save_dir, 'processed_samples.txt')
    processed_samples_patches = set()
    processed_samples_seg = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.endswith('_patches\n'):
                    processed_samples_patches.add(line.strip())
                elif line.endswith('_seg\n'):
                    processed_samples_seg.add(line.strip())

    for name in ['train', 'test', 'validate']:  # ['train', 'test', 'validate']

        X_combo = None
        X_combo_seg = None

        # Read the cellcutter input csv file
        csv_path = os.path.join(cellcutter_input_path, f'{name}_raw.csv')
        csv = pd.read_csv(csv_path)
        csv['Sample'] = csv['Sample'].astype(str)

        # Zip store paths
        zip_path = os.path.join(
            save_dir, f'{name}_patches_{config.window_size}.zip'
        )
        zip_path_seg = os.path.join(
            save_dir, f'{name}_patches_{config.window_size}_seg.zip'
        )

        # Open zip stores (write if new, append if existing)
        if os.path.exists(zip_path):
            zip_store = zarr.ZipStore(zip_path, mode='a')
            X_combo = zarr.open(zip_store, mode='a')

        if os.path.exists(zip_path_seg):
            zip_store_seg = zarr.ZipStore(zip_path_seg, mode='a')
            X_combo_seg = zarr.open(zip_store_seg, mode='a')

        # Loop through each sample in the csv file
        for sample, group in csv.groupby('Sample'):
                
            # Remove any residual sample-specific temp csv files
            files_to_remove = [
                f'{name}_{sample}_patches_{config.window_size}.zip',
                f'{name}_{sample}_patches_{config.window_size}_seg.zip'
            ]
            for file in files_to_remove:
                try:
                    os.remove(os.path.join(save_dir, file))
                except FileNotFoundError:
                    pass
                
            # Save a temp copy of sample data for cellcutter to read
            temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            temp_csv_path = temp_csv.name
            temp_csv.close()
            group.to_csv(temp_csv_path, index=False)
            
            ###################################################################
            
            # Run cellcutter if sample was not already processed for patches
            checkpoint_key_patches = f'{name}_{sample}_patches'
            
            if checkpoint_key_patches not in processed_samples_patches:

                logger.info(f'Cutting data for {checkpoint_key_patches}...')
                print()  

                # Handle paths to TIFFs, masks, and outlines
                if os.path.exists(os.path.join(config.tif_path, f"{sample}.ome.tif")):
                    # multi-tissue input
                    tif_path = os.path.join(config.tif_path, f"{sample}.ome.tif")
                else:
                    # single-tissue input
                    tif_path = config.tif_path
    
                if os.path.exists(os.path.join(config.mask_path, f"{sample}.ome.tif")):
                    # multi-tissue input
                    mask_path = os.path.join(config.mask_path, f"{sample}.ome.tif")
                else:
                    # single-tissue input
                    mask_path = config.mask_path
                
                if os.path.exists(os.path.join(config.outlines_path, f"{sample}.ome.tif")):
                    # multi-tissue input
                    outlines_path = os.path.join(config.outlines_path, f"{sample}.ome.tif")
                else:
                    # single-tissue input
                    outlines_path = config.outlines_path

                # Check pixel size of image
                ome = ome_types.from_tiff(tif_path)
                num_tif_channels = len(ome.images[0].pixels.channels)
                pixel_size_microns = (
                    ome.images[0].pixels.physical_size_x_quantity.to('micron')
                )
                logger.info(f'Physical pixel size = {pixel_size_microns}')
                print()

                pattern = os.path.join(
                    artifact_detection_path, f'{sample}_mask_*.tif'
                )

                matches = glob.glob(pattern)
                if not matches:
                    mask = False
                    logger.info("Artifact mask file does not exist, cutting patches without QC")
                    print()
                    channels_to_cut = marker_channel_numbers 
                else:
                    mask = True
                    mask_path = matches[0]
                    tif_path = append_artifact_mask_to_tif(
                        tif_path, mask_path
                    )
                    channels_to_cut = (
                        marker_channel_numbers + [str(num_tif_channels + 1)]
                    )

                # Run cellcutter to generate image patches
                run(
                    ["cut_cells", "-z", "-f", 
                     "--window-size", str(config.window_size),
                     "--cells-per-chunk", str(config.cells_per_chunk),
                     "--cache-size", str(config.cache_size_cellcutter), 
                     str(tif_path),
                     str(mask_path),
                     str(temp_csv_path),
                     os.path.join(
                         save_dir, 
                         f"{name}_{sample}_patches_{config.window_size}.zip"),
                     "--channels",  
                     ] + channels_to_cut
                )

                # Read cellcutter output image patches
                cellcutter_output_path = os.path.join(
                    save_dir,
                    f'{name}_{sample}_patches_{config.window_size}.zip'
                )
                X = zarr.open(zarr.ZipStore(cellcutter_output_path), mode='r')

                if X_combo is None:
                    # If array doesn't exist, create new one
                    zip_store = zarr.ZipStore(zip_path, mode='w')
                    X_combo = zarr.empty(
                        shape=(X.shape[0], 0, X.shape[2], X.shape[3]),
                        chunks=(X.chunks[0], X.chunks[1],
                                X.shape[2], X.shape[3]),
                        compressor=X.compressor,
                        dtype=X.dtype,
                        store=zip_store
                    )

                # Append new patches
                X_combo.append(X, axis=1)  # append along axis 1

                # After successful path processing, add to checkpoint file
                with open(checkpoint_file, 'a') as f:
                    f.write(f'{checkpoint_key_patches}\n')
                processed_samples_patches.add(checkpoint_key_patches)

            ###################################################################
            
            # Run cellcutter if sample not already processed for seg outlines
            checkpoint_key_seg = f'{name}_{sample}_seg'
            
            if checkpoint_key_seg not in processed_samples_seg:

                logger.info(f'Cutting data for {checkpoint_key_seg}...')
                print()

                # Run cellcutter to generate seg outlines
                run(
                    ["cut_cells", "-z", "-f", 
                     "--window-size", str(config.window_size),
                     "--cells-per-chunk", str(config.cells_per_chunk),
                     "--cache-size", str(config.cache_size_cellcutter),
                     str(outlines_path), 
                     str(mask_path),
                     str(temp_csv_path),
                     os.path.join(
                         save_dir, 
                         f"{name}_{sample}_patches_{config.window_size}_seg.zip"
                     ),
                     "--channels", "1"
                     ]
                )
                
                # Read cellcutter output segmentation outlines
                cellcutter_output_path_seg = os.path.join(
                    save_dir,
                    f'{name}_{sample}_patches_{config.window_size}_seg.zip'
                )
                X_seg = zarr.open(
                    zarr.ZipStore(cellcutter_output_path_seg), mode='r'
                )

                if X_combo_seg is None:
                    # If array doesn't exist, create new one
                    zip_store_seg = zarr.ZipStore(zip_path_seg, mode='w')
                    X_combo_seg = zarr.empty(
                        shape=(X_seg.shape[0], 0, 
                               X_seg.shape[2], X_seg.shape[3]),
                        chunks=(X_seg.chunks[0], X_seg.chunks[1],
                                X_seg.chunks[2], X_seg.chunks[3]),
                        compressor=X_seg.compressor,
                        dtype=X_seg.dtype,
                        store=zip_store_seg
                    )

                # Append new patches
                X_combo_seg.append(X_seg, axis=1)  # append along axis 1

                # After successful outlines processing, add to checkpoint file
                with open(checkpoint_file, 'a') as f:
                    f.write(f'{checkpoint_key_seg}\n')
                processed_samples_seg.add(checkpoint_key_seg)

            try:
                # Remove sample-specific zip stores and temp files
                os.remove(
                    os.path.join(
                        save_dir,
                        f"{name}_{sample}_patches_{config.window_size}.zip")
                )
                os.remove(
                    os.path.join(
                        save_dir,
                        f"{name}_{sample}_patches_{config.window_size}_seg.zip")
                )
                os.remove(temp_csv_path)
                if mask is True:
                    os.remove(tif_path)
            except FileNotFoundError:
                pass

        ########################################################################
        # Filter combined zarrs to remove patches with artifacts
        if config.AAD:
            masks = X_combo[-1]  # artifact mask channel is last channel
            remove_mask = (masks > 1).any(axis=(1, 2))
            keep_mask = ~remove_mask
            X_combo_qc = X_combo.oindex[0:-1, keep_mask, :, :]
            X_combo_qc_seg = X_combo_seg.oindex[:, keep_mask, :, :]
            csv = csv.iloc[keep_mask].reset_index(drop=True)
        else:
            X_combo_qc = X_combo
            X_combo_qc_seg = X_combo_seg
        
        # Create new filtered csv and zarr arrays
        csv.to_csv(
            os.path.join(cellcutter_input_path, f'{name}_qc.csv'), index=False
        )

        filtered_zip_path = os.path.join(
            save_dir, f'{name}_patches_{config.window_size}_qc.zip'
        )
        filtered_store = zarr.ZipStore(filtered_zip_path, mode='w')
        
        filtered_zip_path_seg = os.path.join(
            save_dir, f'{name}_patches_{config.window_size}_qc_seg.zip'
        )
        filtered_store_seg = zarr.ZipStore(filtered_zip_path_seg, mode='w')

        zarr.array(
            X_combo_qc, store=filtered_store, chunks=X_combo.chunks,
            compressor=X_combo.compressor
        )
        zarr.array(
            X_combo_qc_seg, store=filtered_store_seg,
            chunks=X_combo_seg.chunks, compressor=X_combo_seg.compressor
        )

        filtered_store.close()
        filtered_store_seg.close()

        # Close the original collated zip stores
        if X_combo is not None:
            zip_store.close()
        if X_combo_seg is not None:
            zip_store_seg.close()

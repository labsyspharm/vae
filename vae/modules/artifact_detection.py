import os
import sys
import types
import logging

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import zarr
import ome_types
from tifffile import imwrite
from subprocess import run

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from ..utils import log_banner, log_multiline

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# log_multiline(logger.info, pd.DataFrame().to_string(index=False))
# log_banner(logger.info, 'Boolean classifications')


# Define the AAD model class used to generate the saved model
class ArtifactSegmentationModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name,
            in_channels=in_channels, classes=out_classes, **kwargs
        )
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer(
            "std", torch.tensor(params["std"]).mean().view(1, 1, 1, 1)
        )
        self.register_buffer(
            "mean", torch.tensor(params["mean"]).mean().view(1, 1, 1, 1)
        )

        # For image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True
        )

    def forward(self, image):
        # Normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        mask = batch["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        metrics = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        if stage == "train":
            self.training_step_outputs.append(metrics)
        elif stage == "test":
            self.test_step_outputs.append(metrics)
        elif stage == "valid":
            self.validation_step_outputs.append(metrics)
        return metrics

    def shared_on_epoch_end(self, stage):
        if stage == "train":
            outputs = self.training_step_outputs
        elif stage == "test":
            outputs = self.test_step_outputs
        elif stage == "valid":
            outputs = self.validation_step_outputs
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        loss = torch.cat([x["loss"].reshape(1) for x in outputs])
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro"
        )
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_tp": tp.sum(),
            f"{stage}_tn": tn.sum(),
            f"{stage}_fp": fp.sum(),
            f"{stage}_fn": fn.sum(),
            f"{stage}_loss": loss.mean()
        }
        self.log_dict(metrics, prog_bar=True)
        if stage == "train":
            self.training_step_outputs.clear()
        elif stage == "test":
            self.test_step_outputs.clear()
        elif stage == "valid":
            self.validation_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        return self.shared_on_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.shared_on_epoch_end("valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        return self.shared_on_epoch_end("test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


class ChannelPreprocessing():

    def __init__(self, z, channel):
        self.img_tiles = z[channel]

    def __len__(self):
        return len(self.img_tiles)

    def __getitem__(self, idx):
        img_tile = self.img_tiles[idx:idx + 1, :, :]  # to return 1HW
        tile_max, tile_min = img_tile.max(), img_tile.min()
        img_tile = (
            (img_tile - tile_min) / (tile_max - tile_min)
        ).astype('float16')
        print(img_tile.shape)

        return {'image': img_tile}


def load_AAD_model(model_path):
    """
    This function loads a previously saved AAD model. Since the original model
    class is defined in __main__ module we need to temporarily assign the
    model class defined here (class ArtifactSegmentationModel) to the main
    module of the morphaeus pipeline to get access to the saved model. 
    We then revert back to the __main__ module initialized by the pipeline.
    """

    if not os.path.exists(model_path):
        logger.error(f"Model file does not exist: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Save the original __main__ module
    orig_main = sys.modules.get('__main__')
    
    # Create/modify a custom __main__ module
    custom_main = types.ModuleType('__main__')
    
    # Copy all globals to the custom main module
    for key, value in globals().items():
        setattr(custom_main, key, value)
    
    # Set ArtifactSegmentationModel in the custom main module
    custom_main.ArtifactSegmentationModel = ArtifactSegmentationModel
    
    # Replace the pipeline __main__ module temporarily
    sys.modules['__main__'] = custom_main
    
    try:
        # Load saved AAD model
        model = torch.load(model_path)
        logger.info("AAD model loaded successfully!")
        print()
        return model
    except Exception as e:
        logger.error(f"Failed to load AAD model: {e}")
        logger.error(
            "Please check if the model file exists and is accessible."
        )
        print()
        raise
    finally:
        # Restore the original __main__ module
        if orig_main:
            sys.modules['__main__'] = orig_main


def stitch_tiles(tiles, num_x, num_y, tile_size):
    num_y, num_x = num_x, num_y
    tiles_reshaped = tiles.reshape(num_x, num_y, tile_size, tile_size)
    full_image = np.zeros((tile_size * num_x, tile_size * num_y))
    for i in range(num_x):
        for j in range(num_y):
            full_image[i * tile_size: (i + 1) * tile_size, 
                       j * tile_size: (j + 1) * tile_size] = (
                tiles_reshaped[i, j]
            )
    return full_image


def DETECT_ARTIFACTS(config):

    if config.AAD:

        if not os.path.isfile(
           os.path.join(config.output_path,
                        'checkpoints/DETECT_ARTIFACTS.txt')):
            print()
            #######################################################################
            # I/O

            aad_window_size = 2048

            # Use config.yml parameter for model path or fall back to a default
            model_path = getattr(
                config, 'AAD_model_path',  # AAD_model_path is config param name 
                '<A DEFAULT MODEL PATH HERE>'
            )
            model = load_AAD_model(model_path)
            
            save_dir = os.path.join(
                config.output_path, '2_artifact_detection'
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            #######################################################################
            # Crop 2048 x 2048 multi-channel tiles from sample TIFFs
            # (tile size must match those used during AAD model training).
            
            markers = pd.read_csv(config.markers_path)

            # Get the channel numbers for markers in config.tif_channels
            marker_channel_numbers = []
            for i in config.tif_channels:
                id = markers['channel_number'][
                    markers['marker_name'] == i].values[0]
                marker_channel_numbers.append(str(id))
            
            # Add checkpoint tracking file
            checkpoint_file = os.path.join(save_dir, 'processed_samples.txt')
            processed_samples = set()
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    for line in f:
                        if line.endswith('_tiles\n'):
                            processed_samples.add(line.strip())

            # Read the cellcutter input csv file
            csv_path = config.csv_path
            csv = pd.read_parquet(csv_path)
            csv['Sample'] = csv['Sample'].astype(str)

            # Loop through samples in the csv file
            for sample, group in csv.groupby('Sample'):

                # Run cellcutter if sample was not already processed for patches
                checkpoint_key_tiles = f'{sample}_tiles'

                if checkpoint_key_tiles not in processed_samples:

                    logger.info(f'Cutting data for sample {sample}...')
                    print()  

                    # Check pixel size of image (might want to move this elsewhere)
                    ome = ome_types.from_tiff(
                        os.path.join(config.tif_path, f'{sample}.ome.tif')
                    )
                    pixel_size_microns = (
                        ome.images[0].pixels.physical_size_x_quantity.to('micron')
                    )
                    logger.info(f'Physical pixel size = {pixel_size_microns}')
                    print()      

                    # Run cellcutter to generate image patches
                    run(
                        ["cut_tiles", "-z", "-f",
                         "--tiles-per-chunk", str(config.tiles_per_chunk),
                         "--cache-size", str(config.cache_size_cellcutter), "--save-metadata", 
                         os.path.join(
                                      save_dir,
                                      f"{sample}_meta_{aad_window_size}.csv"),
                         str(os.path.join(
                             config.tif_path, f"{sample}.ome.tif" if 
                             os.path.exists(
                                 os.path.join(config.tif_path,
                                              f"{sample}.ome.tif"))
                             else f"{sample}.tif"
                         )),
                         str(aad_window_size), 
                         os.path.join(
                             save_dir, 
                             f"{sample}_tiles_{aad_window_size}.zip"),
                         "--channels",  
                         ] + marker_channel_numbers
                    )

                print()
                
                ###################################################################
                # run AAD model on channel-specific image tiles (2048 x 2048) 

                meta_path = os.path.join(
                    save_dir,
                    f"{sample}_meta_{aad_window_size}.csv"
                )
                meta = pd.read_csv(meta_path)
                
                zarr_path = os.path.join(
                    save_dir, f"{sample}_tiles_{aad_window_size}.zip"
                )
                store = zarr.ZipStore(zarr_path)
                z = zarr.open(store=store, mode='r')
                
                rows = meta['Y_start'].nunique()
                cols = meta['X_start'].nunique()
                
                # Initialize mask to append channel artifact predictions
                mask = np.zeros(shape=(2048 * rows, 2048 * cols), dtype=np.uint8)
                
                for e, ch in enumerate(config.tif_channels):
                    if ch in config.tif_channels:  # Allowing for debugging
                        logger.info(
                            f'Predicting artifacts in {ch} '
                            f'channel of sample {sample}'
                        )
             
                        tiles = ChannelPreprocessing(z, channel=e)
                        
                        tiles_loader = DataLoader(
                            tiles, batch_size=16, shuffle=False, num_workers=4
                        )
                        
                        pr_masks = []
                        for batch_idx, batch in tqdm(enumerate(iter(tiles_loader))):
                            with torch.no_grad():
                                model.eval()
                                logits = model(batch["image"])
                            pr_mask = logits.sigmoid()
                            pr_masks.append(pr_mask)

                        pr_stitched = stitch_tiles(
                            torch.cat(pr_masks, dim=0).squeeze(), cols, rows, 2048
                        ).astype(np.uint8)

                        # Visualize artifact predictions on channel image
                        fig, ax = plt.subplots(figsize=(7, 7))
                        img_stitched = stitch_tiles(z[e], cols, rows, 2048)
                        ax.imshow(img_stitched)
                        ax.imshow(pr_stitched, alpha=0.5)
                        fig.savefig(
                            os.path.join(save_dir, f'{sample}_{ch}.png')
                        )
                        plt.close(fig)

                        mask += pr_stitched

                        print()

                imwrite(
                    os.path.join(save_dir, f'{sample}_mask_{aad_window_size}.tif'),
                    mask, tile=(aad_window_size, aad_window_size), compression='zlib'
                )

                # After successful path processing, add to checkpoint file
                with open(checkpoint_file, 'a') as f:
                    f.write(f'{checkpoint_key_tiles}\n')
                processed_samples.add(checkpoint_key_tiles)

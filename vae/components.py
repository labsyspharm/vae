import logging
import functools

from vae.modules.generate_cellcutter_input import GENERATE_CELLCUTTER_INPUT
from vae.modules.artifact_detection import DETECT_ARTIFACTS
from vae.modules.run_cellcutter import RUN_CELLCUTTER
from vae.modules.generate_image_gallery import GENERATE_IMAGE_GALLERY
from vae.modules.remove_background import REMOVE_BACKGROUND
from vae.modules.train_vae import TRAIN_VAE
from vae.modules.encode_images import ENCODE_IMAGES
from vae.modules.saliency_map import SALIENCY_MAP


def module(func):
    """
    Annotation for pipeline module functions.

    This function adds the given function to the registry list. It also wraps
    the given function to log a pre/post-call banner.

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("=" * 70)
        logger.info("RUNNING MODULE: %s", func.__name__)
        result = func(*args, **kwargs)
        logger.info("=" * 70)
        logger.info("")
        return result
    pipeline_modules.append(wrapper)
    pipeline_module_names.append(wrapper.__name__)
    return wrapper


logger = logging.getLogger(__name__)

# Pipeline module order, to be filled in by the @module decorator.
pipeline_modules = []
pipeline_module_names = []

module(GENERATE_CELLCUTTER_INPUT)
module(DETECT_ARTIFACTS)
module(RUN_CELLCUTTER)
module(GENERATE_IMAGE_GALLERY)
module(REMOVE_BACKGROUND)
module(TRAIN_VAE)
module(ENCODE_IMAGES)
module(SALIENCY_MAP)

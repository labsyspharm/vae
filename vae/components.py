import logging
import functools

from vae.modules.generate_cellcutter_input import GENERATE_CELLCUTTER_INPUT
from vae.modules.run_cellcutter import RUN_CELLCUTTER
from vae.modules.generate_img_gallery import GENERATE_IMAGE_GALLERY
from vae.modules.feature_preprocessing_selections import MAKE_FEATURE_PROCESSING_SELECTIONS
from vae.modules.train_vae import TRAIN_VAE
from vae.modules.encode_imgs import ENCODE_IMAGES


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
module(RUN_CELLCUTTER)
module(GENERATE_IMAGE_GALLERY)
module(MAKE_FEATURE_PROCESSING_SELECTIONS)
module(TRAIN_VAE)
module(ENCODE_IMAGES)

"""
Visual inspection of resizing methods.
"""
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np

from mrcnn.config import mrcnn_config
from mrcnn.utils.utils import resize_image
from samples.nucleus_dataset_handler import NucleusDatasetHandler
from tools.config import Config


logging.basicConfig(stream=sys.stderr, level=logging.INFO)

CONFIG_FILES = ['./samples/nuclei_config.yml',
                './samples/nuclei_config_inference.yml']


def image_resize_test():
    logging.info('Testing image_resize...')

    mrcnn_config.init_config(CONFIG_FILES)

    dataset_train = NucleusDatasetHandler(Config.DATASET_PATH, 'train')

    for img in dataset_train.images:
        new_img, img_metas = resize_image(
            img,
            Config.IMAGE.MIN_DIM,
            Config.IMAGE.MAX_DIM,
            Config.IMAGE.MIN_SCALE,
            Config.IMAGE.RESIZE_MODE
        )
        print(img_metas)
        plt.imshow(new_img.astype(np.uint8))
        plt.show()

    return 0


if __name__ == '__main__':
    image_resize_test()

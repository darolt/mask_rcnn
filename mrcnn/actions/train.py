
import logging
import os

from imgaug import augmenters as iaa

from tools.config import Config


def train(model, dataset_train, dataset_val):
    """Train the model."""
    Config.dump(os.path.join(model.log_dir, 'config.yml'))
    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    logging.info('Train network heads')
    model.fit(dataset_train, dataset_val,
              Config.TRAINING.LEARNING.RATE,
              20, 'heads', augmentation=augmentation)

    logging.info('Train all layers')
    model.fit(dataset_train, dataset_val,
              Config.TRAINING.LEARNING.RATE,
              40, 'all', augmentation=augmentation)

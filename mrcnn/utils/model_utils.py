
import datetime
import logging
import os
import re

import torch

from tools.config import Config


def load_weights(model, filepath, exclude=False):
    """Modified version of the correspoding Keras function with
    the addition of multi-GPU support and the ability to exclude
    some layers from loading.
    exlude: list of layer names to excluce
    """
    # Load weights
    logging.info(f"Loading weights from {filepath}")
    if os.path.exists(filepath):
        state_dict = torch.load(filepath)
        if exclude:
            state_dict = {key: value for key, value in state_dict.items()
                          if key not in exclude}
        model.load_state_dict(state_dict, strict=False)
    else:
        print('Weight file not found ...')

    # Update the log directory
    set_log_dir(model, filepath)
    if not os.path.exists(model.log_dir):
        os.makedirs(model.log_dir)


def find_last(model):
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Returns:
        log_dir: The directory where events and weights are saved
        checkpoint_path: the path to the last checkpoint file
    """
    # Get directory names. Each directory corresponds to a model
    dir_names = next(os.walk(model.model_dir))[1]
    key = Config.NAME.lower()
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        return None, None
    # Pick last directory
    dir_name = os.path.join(model.model_dir, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        return dir_name, None
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return dir_name, checkpoint


def set_log_dir(model, model_path=None):
    """Sets the model log directory and epoch counter.

    model_path: If None, or a format different from what this code uses
        then set a new log directory and start epochs from 0. Otherwise,
        extract the log directory and the epoch counter from the file
        name.
    """
    # Set date and epoch counter as if starting a new model
    model.epoch = 0
    now = datetime.datetime.now()

    # If we have a model path with date and epochs use them
    if model_path:
        # Continue from we left of. Get epoch and date from the file name
        # A sample model path might look like:
        # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.pth
        regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
        m = re.match(regex, model_path)
        if m:
            now = datetime.datetime(int(m.group(1)), int(m.group(2)),
                                    int(m.group(3)), int(m.group(4)),
                                    int(m.group(5)))
            model.epoch = int(m.group(6))

    # Directory for training logs
    model.log_dir = os.path.join(model.model_dir, "{}{:%Y.%m.%d_%H:%M}".format(
        Config.NAME.lower(), now))

    # Path to save after each epoch. Include placeholders that get
    # filled by Keras.
    checkpoint_file = "mask_rcnn_"+Config.NAME.lower()+"_{}.pth"

    model.checkpoint_path = os.path.join(model.log_dir, checkpoint_file)

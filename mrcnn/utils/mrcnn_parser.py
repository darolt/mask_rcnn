
import argparse
import os

from mrcnn.utils.model_utils import find_last
from tools.config import Config


class MRCNNParser(argparse.ArgumentParser):

    def __init__(self, description, root_dir):
        super().__init__(description)

        # Path to trained weights file
        coco_model_path = os.path.join(root_dir, "mask_rcnn_coco.pth")

        default_logs_dir = os.path.join(root_dir, "logs")
        self.add_argument("command",
                          metavar="<command>",
                          choices=['train', 'submit'],
                          help="'train' or 'submit'")
        self.add_argument('--dataset', required=False,
                          metavar="/path/to/coco/",
                          help='Directory of the dataset')
        self.add_argument('--model', required=False,
                          metavar="/path/to/weights.pth",
                          help="Path to weights .pth file or 'coco'")
        self.add_argument('--logs', required=False,
                          default=default_logs_dir,
                          metavar="/path/to/logs/",
                          help='Logs and checkpoints directory (default=logs/)')
        self.add_argument('--dev', required=False,
                          default=0, type=int,
                          help='CUDA current device.')
        self.add_argument('--debug', required=False,
                          type=int, help='Turn on GPU profiler.')
        self.add_argument('--debug_function', required=False,
                          help='name of the function to be debbuged.')
        self.args = self.parse_args()  # pylint: disable=C0103

        self.display()

        if self.args.model:
            if self.args.model.lower() == "coco":
                self.args.model = coco_model_path
            elif self.args.model.lower() == "last":
                # Find last trained weights
                # TODO: fix this
                self.args.model = find_last(model)[1]
            elif self.args.model.lower() == "imagenet":
                # Start from ImageNet trained weights
                self.args.model = Config.IMAGENET_MODEL_PATH
            else:
                self.args.model = self.args.model
        else:
            self.args.model = ""

    def display(self):
        print(f"Command: {self.args.command}")
        print(f"Model: {self.args.model}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Logs: {self.args.logs}")
        print(f"Debug: {self.args.debug}")
        print(f"Debug function: {self.args.debug_function}")

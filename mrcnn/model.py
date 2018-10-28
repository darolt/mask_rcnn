"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import datetime
import os
import re
import gc
import sys
import psutil

import numpy as np
import mrcnn.config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from mrcnn import utils
from mrcnn.anchors import generate_pyramid_anchors
from mrcnn.utils import MRCNNOutput, RPNOutput, RPNTarget,\
                        MRCNNGroundTruth, get_empty_mrcnn_out
from mrcnn.proposal import proposal_layer
from mrcnn.detection import detection_layer, detection_layer2
from mrcnn.dataset import Dataset
from mrcnn.losses import Losses, compute_losses, compute_iou_loss
from mrcnn import visualize
from mrcnn.resnet import ResNet
from mrcnn.rpn import RPN
from mrcnn.fpn import FPN, Classifier, Mask
from mrcnn.detection_target import detection_target_layer


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())


def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)

############################################################
#  MaskRCNN Class
############################################################


class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """
    # Pre-defined layer regular expressions
    LAYER_REGEX = {
        # all layers but the backbone
        "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        # From a specific Resnet stage and up
        "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        # All layers
        "all": ".*",
    }

    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

    def build(self, config):
        """Build Mask R-CNN architecture.
        """
        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        resnet = ResNet("resnet101", stage5=True).float()

        C1, C2, C3, C4, C5 = resnet.stages()

        # Top-down Layers
        # TODO: add assert to verify feature map sizes match what's in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256).float()
        self.fpn.to(mrcnn.config.DEVICE)

        # Generate Anchors
        anchors = generate_pyramid_anchors(
                    config.RPN_ANCHOR_SCALES,
                    config.RPN_ANCHOR_RATIOS,
                    config.BACKBONE_SHAPES,
                    config.BACKBONE_STRIDES,
                    config.RPN_ANCHOR_STRIDE)
        anchors = np.broadcast_to(anchors, (2,) + anchors.shape)
        self.anchors = torch.from_numpy(anchors).float()
        self.anchors = self.anchors.to(mrcnn.config.DEVICE)

        # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS),
                       config.RPN_ANCHOR_STRIDE, 256).float()

        # FPN Classifier
        self.classifier = Classifier(256, config.POOL_SIZE,
                                     config.IMAGE_SHAPE, config.NUM_CLASSES).float()

        # FPN Mask
        self.mask = Mask(256, config.MASK_POOL_SIZE,
                         config.IMAGE_SHAPE, config.NUM_CLASSES).float()

        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.detach().zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.detach().fill_(1)
                m.bias.detach().zero_()
            elif isinstance(m, nn.Linear):
                m.weight.detach().normal_(0, 0.01)
                m.bias.detach().zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False
            else:
                param[1].requires_grad = True

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)),
                                        int(m.group(3)), int(m.group(4)),
                                        int(m.group(5)))
                self.epoch = int(m.group(6))

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%d_%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get
        # filled by Keras.
        checkpoint_file = "mask_rcnn_"+self.config.NAME.lower()+"_{}.pth"

        self.checkpoint_path = os.path.join(self.log_dir, checkpoint_file)

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            if exclude:
                state_dict = {key: value for key, value in state_dict.items()
                              if key not in exclude}
            self.load_state_dict(state_dict, strict=False)
        else:
            print("Weight file not found ...")

        # Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def detect(self, images):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = utils.mold_inputs(images,
                                                                self.config)

        # Convert images to torch tensor
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2))

        # To GPU
        molded_images = molded_images.to(mrcnn.config.DEVICE)

        # Run object detection
        with torch.no_grad():
            detections, mrcnn_mask = self.predict(molded_images, image_metas,
                                                  mode='inference')

        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                utils.unmold_detections(detections[i], mrcnn_mask[i],
                                        image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    @staticmethod
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def predict(self, molded_images, image_metas, mode, gt=None):

        if mode not in ['inference', 'training']:
            raise ValueError(f"mode {mode} not accepted.")

        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

        # Set batchnorm always in eval mode during training
        self.apply(self.set_bn_eval)

        # Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_deltas = outputs
        rpn_out = RPNOutput(rpn_class_logits, rpn_deltas)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = (
            self.config.POST_NMS_ROIS_TRAINING if mode == "training"
            else self.config.POST_NMS_ROIS_INFERENCE)

        batch_size = rpn_class.size()[0]
        anchors = (
            self.anchors if batch_size > 1 else self.anchors[0].unsqueeze(0)
        )
        rpn_rois = proposal_layer([rpn_class, rpn_deltas],
                                  proposal_count=proposal_count,
                                  nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                  anchors=anchors,
                                  config=self.config).float()

        h, w = self.config.IMAGE_SHAPE[:2]
        scale = torch.from_numpy(np.array([h, w, h, w])).float()
        scale = scale.to(mrcnn.config.DEVICE)
        if mode == 'inference':
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_feature_maps_batch = [x[0].unsqueeze(0)
                                        for x in mrcnn_feature_maps]
            mrcnn_class_logits, mrcnn_class, mrcnn_deltas = \
                self.classifier(mrcnn_feature_maps_batch, rpn_rois[0])

            # Detections output is
            # [batch, num_detections, (y1, x1, y2, x2, class_id, score)]
            # in image coordinates
            detections = detection_layer(self.config, rpn_rois, mrcnn_class,
                                         mrcnn_deltas, image_metas)

            detection_boxes = detections[:, :4]/scale
            detection_boxes = detection_boxes.unsqueeze(0)
            # Create masks for detections
            mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)

            # Add back batch dimension
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)

            return (detections, mrcnn_mask)

        elif mode == 'training':
            # Normalize coordinates
            gt_boxes = gt.boxes
            gt.boxes = gt.boxes / scale

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            mrcnn_targets, mrcnn_outs, rois = [], [], []
            precisions = torch.zeros((rpn_rois.size()[0]), dtype=torch.float)
            for i in range(0, rpn_rois.size()[0]):
                rois_, mrcnn_target = detection_target_layer(
                    rpn_rois[i], gt.class_ids[i], gt.boxes[i], gt.masks[i],
                    self.config)

                if rois_.nelement() == 0:
                    mrcnn_out = get_empty_mrcnn_out().to(mrcnn.config.DEVICE)
                    mrcnn_class = torch.FloatTensor().to(mrcnn.config.DEVICE)
                    print('Rois size is empty')
                else:
                    print('Not empty')
                    # Network Heads
                    # Proposal classifier and BBox regressor heads
                    rois_ = rois_.unsqueeze(0)
                    mrcnn_feature_maps_batch = [x[i].unsqueeze(0)
                                                for x in mrcnn_feature_maps]
                    mrcnn_class_logits_, mrcnn_class, mrcnn_deltas_ = \
                        self.classifier(mrcnn_feature_maps_batch, rois_)

                    # Create masks for detections
                    mrcnn_mask_ = self.mask(mrcnn_feature_maps_batch, rois_)
                    mrcnn_out = MRCNNOutput(mrcnn_class_logits_,
                                            mrcnn_deltas_, mrcnn_mask_)

                mrcnn_outs.append(mrcnn_out)
                mrcnn_targets.append(mrcnn_target)

                # print(f"grad: {mrcnn_out.deltas.requires_grad}")
                # print(f"grad: {mrcnn_class.requires_grad}")
                # print(f"grad: {rois_.requires_grad}")
                # prepare mrcnn_out to mAP
                if mrcnn_class.nelement() != 0:
                    detections = detection_layer2(self.config, rois_,
                                                  mrcnn_class, mrcnn_out.deltas,
                                                  image_metas)
                    # print(f"det grad {detections.requires_grad}")
                    # use gt for mAP
                    # call mAP (move to loss)
                    try:
                        # print(f"mrcnn_mask_ {mrcnn_mask_.requires_grad}")
                        precision = compute_iou_loss(gt.masks[i], gt_boxes[i],
                                                     gt.class_ids[i],
                                                     image_metas[i], self.config,
                                                     detections,
                                                     mrcnn_mask_)
                        # print(f"precision {precision.requires_grad}")
                        print(f"precision is {precision}")
                        precisions[i] = precision
                    except utils.NonPositiveAreaError as e:
                        print(str(e))

            return (rpn_out, mrcnn_targets, mrcnn_outs, precisions.mean())

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs,
                    layers, augmentation=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        if layers in MaskRCNN.LAYER_REGEX.keys():
            layers = MaskRCNN.LAYER_REGEX[layers]

        # Data generators
        train_set = Dataset(train_dataset, self.config,
                            augmentation=augmentation)
        train_generator = torch.utils.data.DataLoader(train_set,
                                                      batch_size=2,
                                                      shuffle=True,
                                                      num_workers=4)
        val_set = Dataset(val_dataset, self.config,
                          augmentation=augmentation)
        val_generator = torch.utils.data.DataLoader(val_set,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    num_workers=4)

        # Train
        utils.log(f"\nStarting at epoch {self.epoch+1}. LR={learning_rate}\n")
        self.set_trainable(layers)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters()
                            if param.requires_grad and 'bn' not in name]
        trainables_only_bn = [param for name, param in self.named_parameters()
                              if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn,
             'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch+1, epochs+1):
            utils.log("Epoch {}/{}.".format(epoch, epochs))

            # Training
            train_losses = self.train_epoch(train_generator,
                                            optimizer,
                                            self.config.STEPS_PER_EPOCH)

            # Validation
            with torch.no_grad():
                val_losses = self.valid_epoch(val_generator,
                                              self.config.VALIDATION_STEPS)

            # Statistics
            self.loss_history.append(train_losses.to_list())
            self.val_loss_history.append(val_losses.to_list())
            visualize.plot_loss(self.loss_history,
                                self.val_loss_history,
                                save=True,
                                log_dir=self.log_dir)

            # Save model
            torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    def train_epoch(self, datagenerator, optimizer, steps):
        losses_sum = Losses()

        for step, inputs in enumerate(datagenerator):
            if step == steps:
                break

            # To GPU
            images, image_metas, rpn_target, gt = self.prepare_inputs(inputs)

            optimizer.zero_grad()

            # Run object detection
            outputs = self.predict(images, image_metas, mode='training', gt=gt)

            del images, image_metas, gt

            # Compute losses
            losses_epoch = compute_losses(rpn_target, *outputs[:-1])
            del rpn_target

            # Backpropagation
            #losses_epoch.total.backward()
            if outputs[2][0].deltas.nelement() == 0 or outputs[2][1].deltas.nelement() == 0:
                losses_epoch.total.backward()
            else:
                print('optimizing mAP')
                outputs[-1].backward()

            del outputs

            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()

            # Progress
            utils.printProgressBar(step + 1, steps, losses_epoch)

            # Statistics
            losses_sum = losses_sum + losses_epoch.item()/steps

            del losses_epoch

        return losses_sum

    def valid_epoch(self, datagenerator, steps):
        losses_sum = Losses()

        for step, inputs in enumerate(datagenerator):
            # Break after 'steps' steps
            if step == steps:
                break

            # To GPU
            images, image_metas, rpn_target, gt = self.prepare_inputs(inputs)

            # Run object detection
            outputs = self.predict(images, image_metas, mode='training', gt=gt)

            try:
                if outputs[1][0].class_ids.nelement() == 0:
                    continue
            except IndexError:
                continue

            # Compute losses
            losses_epoch = compute_losses(rpn_target, *outputs)

            # Progress
            utils.printProgressBar(step + 1, steps, losses_epoch)

            # Statistics
            losses_sum = losses_sum + losses_epoch/steps

        return losses_sum

    def prepare_inputs(self, inputs):
        images = inputs[0].to(mrcnn.config.DEVICE)
        image_metas = inputs[1]
        rpn_target = RPNTarget(inputs[2], inputs[3])
        gt = MRCNNGroundTruth(inputs[4], inputs[5], inputs[6])
        rpn_target.to(mrcnn.config.DEVICE)
        gt.to(mrcnn.config.DEVICE)

        return (images, image_metas, rpn_target, gt)

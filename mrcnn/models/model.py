"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import logging
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from mrcnn.data.data_generator import DataGenerator
from mrcnn.functions.losses import Losses, compute_losses
from mrcnn.models.components.anchors import generate_pyramid_anchors
from mrcnn.models.components.classifier_head import Classifier
from mrcnn.models.components.detection import detection_layer
from mrcnn.models.components.detection_target import detection_target_layer
from mrcnn.models.components.fpn import FPN
from mrcnn.models.components.mask_head import Mask
from mrcnn.models.components.proposal import proposal_layer
from mrcnn.models.components.resnet import ResNet
from mrcnn.models.components.rpn import RPN
from mrcnn.structs.mrcnn_ground_truth import MRCNNGroundTruth
from mrcnn.structs.mrcnn_output import MRCNNOutput
from mrcnn.structs.rpn_output import RPNOutput
from mrcnn.structs.rpn_target import RPNTarget
from mrcnn.utils import utils
from mrcnn.utils import visualize
from mrcnn.utils.model_utils import set_log_dir
from mrcnn.utils.progress_bar import ProgressBar
from tools.config import Config
from tools.time_profiling import profilable


class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """
    # Pre-defined layer regular expressions
    _LAYER_REGEX = {
        # all layer`s but the backbone
        "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        # From a specific Resnet stage and up
        "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        # All layers
        "all": ".*",
    }

    def __init__(self, model_dir):
        """
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.model_dir = model_dir
        set_log_dir(self)
        self.build()
        self.initialize_weights()

    def build(self):
        """Build Mask R-CNN architecture.
        """
        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        resnet = ResNet("resnet101", stage5=True).float()

        C1, C2, C3, C4, C5 = resnet.stages()

        # Top-down Layers
        self.fpn = (FPN(C1, C2, C3, C4, C5, out_channels=256)
                    .float().to(Config.DEVICE))

        # Generate Anchors
        self.anchors = generate_pyramid_anchors(
            Config.RPN.ANCHOR.SCALES,
            Config.RPN.ANCHOR.RATIOS,
            Config.BACKBONE.SHAPES,
            Config.BACKBONE.STRIDES,
            Config.RPN.ANCHOR.STRIDE,
            Config.TRAINING.BATCH_SIZE
        ).to(Config.DEVICE)

        # RPN
        self.rpn = RPN(len(Config.RPN.ANCHOR.RATIOS),
                       Config.RPN.ANCHOR.STRIDE, 256).float()

        # FPN Classifier
        self.classifier = Classifier(
            256, Config.HEADS.POOL_SIZE,
            Config.IMAGE.SHAPE, Config.NUM_CLASSES).float()

        # FPN Mask
        self.mask = Mask(256, Config.HEADS.MASK.POOL_SIZE,
                         Config.IMAGE.SHAPE, Config.NUM_CLASSES).float()

        # Fix batch norm layers
        def set_bn_fix(model):
            classname = model.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for parameter in model.parameters():
                    parameter.requires_grad = False

        self.apply(set_bn_fix)
        self.to(Config.DEVICE)

    def initialize_weights(self):
        """Initialize model weights."""
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

    def set_trainable(self, layer_regex):
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

    def detect(self, image):
        """Runs the detection pipeline.

        images: Image

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        # Mold inputs to format expected by the neural network
        molded_image, image_metas = utils.mold_inputs([image])

        # Convert images to torch tensor
        molded_image = (torch.from_numpy(molded_image).float()
                        .permute(0, 3, 1, 2).to(Config.DEVICE))

        # Run object detection
        self.eval()
        self.apply(self._set_bn_eval)
        with torch.no_grad():
            detections, mrcnn_masks = self._predict(
                molded_image,
                Config.PROPOSALS.POST_NMS_ROIS.INFERENCE,
                mode='inference')

        mrcnn_masks = mrcnn_masks.permute(0, 2, 3, 1)
        # Process detections
        result = utils.unmold_detections(
            detections, mrcnn_masks, image_metas)
        return result, image_metas

    @staticmethod
    def _set_bn_eval(model):
        classname = model.__class__.__name__
        if classname.find('BatchNorm') != -1:
            model.eval()

    @profilable
    def _foreground_background_layer(self, molded_images):

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
        rpn_out = RPNOutput(*outputs)

        return mrcnn_feature_maps, rpn_out

    def _inference(self, mrcnn_feature_maps, rpn_rois):
        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_feature_maps_batch = [x[0].unsqueeze(0)
                                    for x in mrcnn_feature_maps]
        _, mrcnn_class, mrcnn_deltas = \
            self.classifier(mrcnn_feature_maps_batch, rpn_rois[0])

        # Detections output is
        # [batch, num_detections, (y1, x1, y2, x2, class_id, score)]
        # in image coordinates
        with torch.no_grad():
            detections = detection_layer(rpn_rois, mrcnn_class,
                                         mrcnn_deltas)

        detection_boxes = detections[:, :4]/Config.RPN.NORM
        detection_boxes = detection_boxes.unsqueeze(0)
        # Create masks for detections
        mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)

        # Add back batch dimension
        detections = detections
        mrcnn_mask = mrcnn_mask

        return detections, mrcnn_mask

    @profilable
    def _predict(self, molded_images, proposal_count,
                 mode='training', gt=None):

        if mode not in ['inference', 'training']:
            raise ValueError(f"mode {mode} not accepted.")

        mrcnn_feature_maps, rpn_out = \
            self._foreground_background_layer(molded_images)

        batch_size = rpn_out.classes.shape[0]
        anchors = (
            self.anchors if batch_size > 1 else self.anchors[0].unsqueeze(0)
        )
        with torch.no_grad():
            rpn_rois = proposal_layer(  # Generate proposals
                rpn_out.classes,
                rpn_out.deltas,
                proposal_count=proposal_count,
                nms_threshold=Config.RPN.NMS_THRESHOLD,
                anchors=anchors)

        if mode == 'inference':
            return self._inference(mrcnn_feature_maps, rpn_rois)
        elif mode == 'training':
            # Normalize coordinates
            gt.boxes = gt.boxes / Config.RPN.NORM

            mrcnn_targets, mrcnn_outs = [], []
            for img_idx in range(0, batch_size):
                with torch.no_grad():
                    rois, mrcnn_target = detection_target_layer(
                        rpn_rois[img_idx], gt.class_ids[img_idx],
                        gt.boxes[img_idx], gt.masks[img_idx])

                if rois.nelement() == 0:
                    mrcnn_out = MRCNNOutput().to(Config.DEVICE)
                    logging.debug('Rois size is empty')
                else:
                    # Network Heads
                    # Proposal classifier and BBox regressor heads
                    rois = rois.unsqueeze(0)
                    mrcnn_feature_maps_batch = [x[img_idx].unsqueeze(0).detach()
                                                for x in mrcnn_feature_maps]
                    mrcnn_class_logits_, _, mrcnn_deltas_ = \
                        self.classifier(mrcnn_feature_maps_batch, rois)

                    # Create masks
                    mrcnn_mask_ = self.mask(mrcnn_feature_maps_batch, rois)

                    mrcnn_out = MRCNNOutput(mrcnn_class_logits_,
                                            mrcnn_deltas_, mrcnn_mask_)

                mrcnn_outs.append(mrcnn_out)
                mrcnn_targets.append(mrcnn_target)

            return rpn_out, mrcnn_targets, mrcnn_outs

    def _get_generators(self, train_dataset, val_dataset, augmentation):
        train_set = DataGenerator(train_dataset, augmentation=augmentation)
        train_generator = torch.utils.data.DataLoader(
            train_set, shuffle=True, batch_size=Config.TRAINING.BATCH_SIZE,
            num_workers=4)
        val_set = DataGenerator(val_dataset, augmentation=augmentation)
        val_generator = torch.utils.data.DataLoader(
            val_set, batch_size=1, shuffle=True, num_workers=4)
        return train_generator, val_generator

    def fit(self, train_dataset, val_dataset, learning_rate, epochs,
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
        if layers in MaskRCNN._LAYER_REGEX.keys():
            layers = MaskRCNN._LAYER_REGEX[layers]

        # Data generators
        train_generator, val_generator = self._get_generators(
            train_dataset, val_dataset, augmentation)

        # Train
        logging.info(f"Starting at epoch {self.epoch+1}. "
                     f"LR={learning_rate}")
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
             'weight_decay': Config.TRAINING.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=Config.TRAINING.LEARNING.MOMENTUM)

        self.train()
        self.apply(self._set_bn_eval)

        loss_history, val_loss_history = [], []
        for epoch in range(self.epoch+1, epochs+1):
            logging.info(f"Epoch {epoch}/{epochs}.")

            # Training
            train_losses = self._train_epoch(train_generator, optimizer)

            # Validation
            with torch.no_grad():
                val_losses = self._validation_epoch(val_generator)

            # Statistics
            loss_history.append(train_losses)
            val_loss_history.append(val_losses)
            visualize.plot_losses(loss_history,
                                  val_loss_history,
                                  log_dir=self.log_dir)

            # Save model
            torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    def _train_epoch(self, datagenerator, optimizer):
        """Trains a single epoch."""
        losses_sum = Losses()
        steps = Config.TRAINING.STEPS_PER_EPOCH
        progress_bar = ProgressBar(steps)

        for step, inputs in enumerate(datagenerator):
            if step == steps:
                break

            # To GPU
            with torch.no_grad():
                images, image_metas, rpn_target, gt = \
                    self._prepare_inputs(inputs)

            optimizer.zero_grad()

            # Run object detection
            rpn_out, mrcnn_targets, mrcnn_outs = \
                self._predict(
                    images,
                    Config.PROPOSALS.POST_NMS_ROIS.TRAINING,
                    gt=gt)
            del images, image_metas, gt

            # Compute losses
            losses = compute_losses(rpn_target, rpn_out,
                                    mrcnn_targets, mrcnn_outs)
            del rpn_target, rpn_out, mrcnn_targets, mrcnn_outs

            # Backpropagation
            losses.total.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)

            optimizer.step()

            progress_bar.print(losses)

            # Statistics
            losses_sum = losses_sum + losses.item()/steps

            del losses

        return losses_sum

    def _validation_epoch(self, datagenerator):
        """Validation step. Usually called with torch.no_grad()."""
        losses_sum = Losses()
        steps = Config.TRAINING.VALIDATION_STEPS
        progress_bar = ProgressBar(steps)

        for step, inputs in enumerate(datagenerator):
            if step == steps:
                break

            # To GPU
            images, _, rpn_target, gt = self._prepare_inputs(inputs)

            # Run object detection
            outputs = self._predict(
                images,
                Config.PROPOSALS.POST_NMS_ROIS.TRAINING,
                gt=gt)

            losses = compute_losses(rpn_target, *outputs)

            progress_bar.print(losses)

            # Statistics
            losses_sum = losses_sum + losses.item()/steps
            del losses

        return losses_sum

    @staticmethod
    def _prepare_inputs(inputs):
        images = inputs[0].to(Config.DEVICE)
        image_metas = inputs[1]
        rpn_target = (RPNTarget(inputs[2], inputs[3])
                      .to(Config.DEVICE))
        gt = (MRCNNGroundTruth(inputs[4], inputs[5], inputs[6])
              .to(Config.DEVICE))

        return (images, image_metas, rpn_target, gt)

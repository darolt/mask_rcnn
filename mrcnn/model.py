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

import numpy as np
import mrcnn.config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from mrcnn import utils
from mrcnn.proposal import proposal_layer
from mrcnn.detection import detection_layer
from mrcnn.dataset import Dataset
from mrcnn.losses import Losses, compute_losses, compute_iou_loss
from mrcnn import visualize
from mrcnn.resnet import ResNet
from mrcnn.rpn import RPN
from mrcnn.fpn import FPN, Classifier, Mask
from roialign.roi_align.crop_and_resize import CropAndResizeFunction


############################################################
#  Detection Target Layer
############################################################


def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every box1 against every box2 without loops.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = torch.zeros(y1.size()[0], requires_grad=False, dtype=torch.float32,
                        device=mrcnn.config.DEVICE)
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:, 0] + b2_area[:, 0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    nans = (iou != iou)
    # TODO check if this impacts gradients
    iou[nans] = -1
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps


def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks,
                           config):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """
    # Handle crowds
    # A crowd box is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = torch.nonzero(gt_class_ids < 0)  # [:, 0]
    if crowd_ix.nelement() != 0:
        crowd_ix = crowd_ix[:, 0]
        non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, 0]
        crowd_boxes = gt_boxes[crowd_ix.detach(), :]
        gt_class_ids = gt_class_ids[non_crowd_ix.detach()]
        gt_boxes = gt_boxes[non_crowd_ix.detach(), :]
        gt_masks = gt_masks[non_crowd_ix.detach(), :]

        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = torch.tensor(proposals.size()[0]*[True],
                                     dtype=torch.uint8,
                                     device=mrcnn.config.DEVICE,
                                     requires_grad=False)

    # Compute overlaps matrix [nb_batches, proposals, gt_boxes]
    # print("Before...")
    # print("Proposals: {}".format(proposals.size()))
    # print("GT_boxes: {}".format(gt_boxes.size()))
    overlaps = bbox_overlaps(proposals, gt_boxes)
    # print("Overlaps {}".format(overlaps.size()))

    # Determine positive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]
    # print("ROI_IOU_max {}".format(roi_iou_max.size()))

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    # print("TRAIN_ROIS_PER_IMAGE: {}".format(config.TRAIN_ROIS_PER_IMAGE))
    # print("ROI_POSITIVE_RATIO: {}".format(config.ROI_POSITIVE_RATIO))
    if torch.nonzero(positive_roi_bool).nelement() != 0:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count].to(mrcnn.config.DEVICE)
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.detach(), :]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.detach(), :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.detach(), :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.detach()]

        # Compute bbox refinement for positive ROIs
        deltas = utils.box_refinement(positive_rois.detach(),
                                      roi_gt_boxes.detach())
        std_dev = torch.from_numpy(config.BBOX_STD_DEV).float()
        std_dev = std_dev.to(mrcnn.config.DEVICE)
        deltas /= std_dev

        # Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment.detach(), :, :]

        # Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = (torch.arange(roi_masks.size()[0]).int()
                   .to(mrcnn.config.DEVICE))
        masks = CropAndResizeFunction(
                    config.MASK_SHAPE[0],
                    config.MASK_SHAPE[1],
                    0)(roi_masks.unsqueeze(1), boxes, box_ids).detach()
        masks = masks.squeeze(1)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = torch.round(masks)
    else:
        positive_count = 0
    # print("positive_count {}".format(positive_count))

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    if torch.nonzero(negative_roi_bool).nelement() != 0 and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count].to(mrcnn.config.DEVICE)
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.detach(), :]
    else:
        negative_count = 0
    # print("negative_count {}".format(negative_count))

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = torch.zeros(negative_count, dtype=torch.int,
                            device=mrcnn.config.DEVICE)
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = torch.zeros(negative_count, 4, dtype=torch.float32,
                            device=mrcnn.config.DEVICE)
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = torch.zeros(negative_count, config.MASK_SHAPE[0],
                            config.MASK_SHAPE[1], dtype=torch.float32,
                            device=mrcnn.config.DEVICE)
        masks = torch.cat([masks, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        roi_gt_class_ids = torch.zeros(negative_count,
                                       device=mrcnn.config.DEVICE)
        deltas = torch.zeros(negative_count, 4, dtype=torch.int,
                             device=mrcnn.config.DEVICE)
        masks = torch.zeros(negative_count, config.MASK_SHAPE[0],
                            config.MASK_SHAPE[1], device=mrcnn.config.DEVICE)
    else:
        rois = torch.tensor([], dtype=torch.float32,
                            device=mrcnn.config.DEVICE)
        roi_gt_class_ids = torch.tensor([], dtype=torch.int,
                                        device=mrcnn.config.DEVICE)
        deltas = torch.tensor([], dtype=torch.float32,
                              device=mrcnn.config.DEVICE)
        masks = torch.tensor([], dtype=torch.float32,
                             device=mrcnn.config.DEVICE)

    # print("masks {}".format(masks.size()))
    return rois, roi_gt_class_ids, deltas, masks


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
        # TODO: add assert to varify feature map sizes match what's in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256).float()
        self.fpn.to(mrcnn.config.DEVICE)

        # Generate Anchors
        anchors = utils.generate_pyramid_anchors(
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
        molded_images, image_metas, windows = utils.mold_inputs(images, self.config)

        # Convert images to torch tensor
        molded_images = (torch.from_numpy(molded_images.transpose(0, 3, 1, 2))
                              .float())

        # To GPU
        molded_images = molded_images.to(mrcnn.config.DEVICE)

        # Run object detection
        with torch.no_grad():
            detections, mrcnn_mask = self.predict([molded_images, image_metas],
                                                  mode='inference')

        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                utils.unmold_detections(detections[i], mrcnn_mask[i],
                                        image.shape, windows[i])
            results.append({
                "rois": final_rois.detach().cpu().numpy(),
                "class_ids": final_class_ids.detach().cpu().numpy(),
                "scores": final_scores.detach().cpu().numpy(),
                "masks": final_masks.detach().cpu().numpy(),
            })
        return results

    def compute_metric(self, testset):
        # Data generators
        testset = Dataset(testset, self.config)
        test_generator = torch.utils.data.DataLoader(testset,
                                                     batch_size=2,
                                                     shuffle=False,
                                                     num_workers=4)

        utils.log(f"\nComputing metric.\n")

        precisions = torch.empty((len(test_generator)),
                                 device=mrcnn.config.DEVICE)
        for idx, inputs in enumerate(test_generator):
            # To GPU
            images = inputs[0].to(mrcnn.config.DEVICE)
            image_metas = inputs[1]
            gt_class_ids = inputs[4].to(mrcnn.config.DEVICE)
            gt_boxes = inputs[5].to(mrcnn.config.DEVICE)
            gt_masks = inputs[6].to(mrcnn.config.DEVICE)

            # Run object detection
            with torch.no_grad():
                outputs = self.predict([images, image_metas, gt_class_ids,
                                       gt_boxes, gt_masks], mode='training')

            # calculate Kaggle metric here
            precision = compute_iou_loss(gt_masks[0], gt_boxes[0], gt_class_ids[0],
                                         image_metas[0], outputs, self.config)
#                             outputs[7][0], outputs[5][0], outputs[3][0],
#                             image_metas[0], outputs[8][0], self.config,
#                             outputs[9][0])
            precisions[idx] = precision

        print(f"Final precision: {precisions.mean()}")

    def predict(self, inputs, mode):
        molded_images = inputs[0]
        image_metas = inputs[1]

        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

        # Set batchnorm always in eval mode during training
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.apply(set_bn_eval)

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
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = (
            self.config.POST_NMS_ROIS_TRAINING
            if mode == "training"
            else self.config.POST_NMS_ROIS_INFERENCE)
        batch_size = rpn_class.size()[0]
        anchors = (
            self.anchors if batch_size > 1 else self.anchors[0].unsqueeze(0)
        )
        rpn_rois = proposal_layer([rpn_class, rpn_bbox],
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
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                self.classifier(mrcnn_feature_maps_batch, rpn_rois[0])

            # Detections output is
            # [batch, num_detections, (y1, x1, y2, x2, class_id, score)]
            # in image coordinates
            detections = detection_layer(self.config, rpn_rois, mrcnn_class,
                                         mrcnn_bbox, image_metas)

            # Convert boxes to normalized coordinates
            # TODO: let DetectionLayer return normalized coordinates to avoid
            #       unnecessary conversions
            detection_boxes = detections[:, :4]/scale

            # Add back batch dimension
            detection_boxes = detection_boxes.unsqueeze(0)

            # Create masks for detections
            mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)

            # Add back batch dimension
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)

            return [detections, mrcnn_mask]

        elif mode == 'training':
            gt_class_ids = inputs[2]
            gt_boxes = inputs[3]
            gt_masks = inputs[4]

            # Normalize coordinates
            gt_boxes = gt_boxes / scale

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            target_class_ids, target_deltas, target_mask = [], [], []
            mrcnn_class_logits, mrcnn_bbox, mrcnn_mask = [], [], []
            mrcnn_probs = []
            rois = []
            for i in range(0, rpn_rois.size()[0]):
                rois_, target_class_ids_, target_deltas_, target_mask_ = \
                    detection_target_layer(rpn_rois[i], gt_class_ids[i],
                                           gt_boxes[i], gt_masks[i],
                                           self.config)

                if rois_.nelement() == 0:
                    mrcnn_class_logits_ = torch.FloatTensor().to(mrcnn.config.DEVICE)
                    mrcnn_bbox_ = torch.FloatTensor().to(mrcnn.config.DEVICE)
                    mrcnn_mask_ = torch.FloatTensor().to(mrcnn.config.DEVICE)
                    mrcnn_probs_ = torch.FloatTensor().to(mrcnn.config.DEVICE)
                    # TODO how to solve this?
                    print('Rois size is empty')
                else:
                    # Network Heads
                    # Proposal classifier and BBox regressor heads
                    rois_ = rois_.unsqueeze(0)
                    mrcnn_feature_maps_batch = [x[i].unsqueeze(0)
                                                for x in mrcnn_feature_maps]
                    mrcnn_class_logits_, mrcnn_probs_, mrcnn_bbox_ = \
                        self.classifier(mrcnn_feature_maps_batch, rois_)

                    # Create masks for detections
                    mrcnn_mask_ = self.mask(mrcnn_feature_maps_batch, rois_)
                mrcnn_mask.append(mrcnn_mask_)
                mrcnn_bbox.append(mrcnn_bbox_)
                mrcnn_probs.append(mrcnn_probs_)
                mrcnn_class_logits.append(mrcnn_class_logits_)
                target_class_ids.append(target_class_ids_)
                target_deltas.append(target_deltas_)
                target_mask.append(target_mask_)
                rois.append(rois_)

            return [rpn_class_logits, rpn_bbox, target_class_ids,
                    mrcnn_class_logits, target_deltas, mrcnn_bbox,
                    target_mask, mrcnn_mask, rois, mrcnn_probs]

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
            images = inputs[0].to(mrcnn.config.DEVICE)
            image_metas = inputs[1]
            rpn_match = inputs[2].to(mrcnn.config.DEVICE)
            rpn_bbox = inputs[3].to(mrcnn.config.DEVICE)
            gt_class_ids = inputs[4].to(mrcnn.config.DEVICE)
            gt_boxes = inputs[5].to(mrcnn.config.DEVICE)
            gt_masks = inputs[6].to(mrcnn.config.DEVICE)

            optimizer.zero_grad()

            # Run object detection
            outputs = self.predict([images, image_metas, gt_class_ids,
                                   gt_boxes, gt_masks], mode='training')

            # Compute losses
            losses_epoch = compute_losses(rpn_match, rpn_bbox, *outputs[:-2])

            # Backpropagation
            losses_epoch.total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()

            # Progress
            utils.printProgressBar(step + 1, steps, losses_epoch)

            # Statistics
            losses_sum = losses_sum + losses_epoch/steps

        return losses_sum

    def valid_epoch(self, datagenerator, steps):
        losses_sum = Losses()

        for step, inputs in enumerate(datagenerator):
            # Break after 'steps' steps
            if step == steps:
                break

            # To GPU
            images = inputs[0].to(mrcnn.config.DEVICE)
            image_metas = inputs[1]
            rpn_match = inputs[2].to(mrcnn.config.DEVICE)
            rpn_bbox = inputs[3].to(mrcnn.config.DEVICE)
            gt_class_ids = inputs[4].to(mrcnn.config.DEVICE)
            gt_boxes = inputs[5].to(mrcnn.config.DEVICE)
            gt_masks = inputs[6].to(mrcnn.config.DEVICE)

            # Run object detection
            outputs = self.predict([images, image_metas, gt_class_ids,
                                    gt_boxes, gt_masks], mode='training')

            try:
                if outputs[2][0].nelement() == 0:
                    continue
            except IndexError:
                continue

            # Compute losses
            losses_epoch = compute_losses(rpn_match, rpn_bbox, *outputs[:-2])

            # Progress
            utils.printProgressBar(step + 1, steps, losses_epoch)

            # Statistics
            losses_sum = losses_sum + losses_epoch/steps

        return losses_sum

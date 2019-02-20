
import torch.nn.functional as F
import torch

from tools.config import Config


class Losses():
    def __init__(self, rpn_class=0.0, rpn_bbox=0.0, mrcnn_class=0.0,
                 mrcnn_bbox=0.0, mrcnn_mask=0.0):
        self.rpn_class = rpn_class
        self.rpn_bbox = rpn_bbox
        self.mrcnn_class = mrcnn_class
        self.mrcnn_bbox = mrcnn_bbox
        self.mrcnn_mask = mrcnn_mask
        self.update_total_loss()

    def item(self):
        return Losses(self.rpn_class.item(), self.rpn_bbox.item(),
                      self.mrcnn_class.item(), self.mrcnn_bbox.item(),
                      self.mrcnn_mask.item())

    def to_list(self):
        return [self.total, self.rpn_class, self.rpn_bbox,
                self.mrcnn_class, self.mrcnn_bbox, self.mrcnn_mask]

    def update_total_loss(self):
        self.total = self.rpn_class + self.rpn_bbox + self.mrcnn_class + \
                     self.mrcnn_bbox + self.mrcnn_mask

    def __truediv__(self, b):
        new_rpn_class = self.rpn_class/b
        new_rpn_bbox = self.rpn_bbox/b
        new_mrcnn_class = self.mrcnn_class/b
        new_mrcnn_bbox = self.mrcnn_bbox/b
        new_mrcnn_mask = self.mrcnn_mask/b
        return Losses(new_rpn_class, new_rpn_bbox, new_mrcnn_class,
                      new_mrcnn_bbox, new_mrcnn_mask)

    def __add__(self, other):
        new_rpn_class = self.rpn_class + other.rpn_class
        new_rpn_bbox = self.rpn_bbox + other.rpn_bbox
        new_mrcnn_class = self.mrcnn_class + other.mrcnn_class
        new_mrcnn_bbox = self.mrcnn_bbox + other.mrcnn_bbox
        new_mrcnn_mask = self.mrcnn_mask + other.mrcnn_mask
        return Losses(new_rpn_class, new_rpn_bbox, new_mrcnn_class,
                      new_mrcnn_bbox, new_mrcnn_mask)


def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0).detach()

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices[:, 0],
                                        indices[:, 1], :]
    anchor_class = anchor_class[indices[:, 0], indices[:, 1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unused bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match == 1).detach()

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices[:, 0], indices[:, 1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    ranges_per_img = torch.empty((indices.shape[0]), dtype=torch.long)
    count = 0
    for img_idx in range(target_bbox.shape[0]):
        nb_elem = torch.nonzero(indices[:, 0] == img_idx).shape[0]
        ranges_per_img[count:count+nb_elem] = torch.arange(nb_elem)
        count += nb_elem
    target_bbox = target_bbox[indices[:, 0], ranges_per_img]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """
    # Loss
    if target_class_ids.nelement() != 0 and pred_class_logits.nelement() != 0:
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = torch.tensor([0], dtype=torch.float32, device=Config.DEVICE)

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if target_class_ids.nelement() != 0:
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.detach()].long()

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[positive_roi_ix[:].detach(), :]
        pred_bbox = pred_bbox[positive_roi_ix[:].detach(),
                              positive_roi_class_ids[:].detach(), :]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = torch.tensor([0], dtype=torch.float32, device=Config.DEVICE)

    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, num_classes, height, width] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.nelement() != 0:
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.detach()].long()

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[positive_ix[:].detach(), :, :]
        y_pred = pred_masks[positive_ix[:].detach(),
                            positive_class_ids[:].detach(),
                            :, :]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = torch.tensor([0], dtype=torch.float32, device=Config.DEVICE)

    return loss


def compute_rpn_losses(rpn_target, rpn_out):

    rpn_class_loss = compute_rpn_class_loss(rpn_target.match,
                                            rpn_out.class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_target.deltas,
                                          rpn_target.match, rpn_out.deltas)
    zero = torch.tensor([0.0], dtype=torch.float32, device=Config.DEVICE)

    return Losses(rpn_class_loss, rpn_bbox_loss, zero.clone(),
                  zero.clone(), zero.clone())


def compute_mrcnn_losses(mrcnn_targets, mrcnn_outs):
    zero = torch.tensor([0.0], dtype=torch.float32, device=Config.DEVICE)
    mrcnn_class_loss = zero.clone()
    mrcnn_bbox_loss = zero.clone()
    mrcnn_mask_loss = zero.clone()

    for mrcnn_target, mrcnn_out in zip(mrcnn_targets, mrcnn_outs):
        mrcnn_class_loss += compute_mrcnn_class_loss(
            mrcnn_target.class_ids, mrcnn_out.class_logits)
        mrcnn_bbox_loss += compute_mrcnn_bbox_loss(
            mrcnn_target.deltas, mrcnn_target.class_ids, mrcnn_out.deltas)
        mrcnn_mask_loss += compute_mrcnn_mask_loss(
            mrcnn_target.masks, mrcnn_target.class_ids, mrcnn_out.masks)

    if len(mrcnn_outs) != 0:
        mrcnn_class_loss /= len(mrcnn_outs)
        mrcnn_bbox_loss /= len(mrcnn_outs)
        mrcnn_mask_loss /= len(mrcnn_outs)

    return Losses(zero.clone(), zero.clone(), mrcnn_class_loss,
                  mrcnn_bbox_loss, mrcnn_mask_loss)


def compute_losses(rpn_target, rpn_out, mrcnn_targets, mrcnn_outs):
    rpn_loss = compute_rpn_losses(rpn_target, rpn_out)
    mrcnn_loss = compute_mrcnn_losses(mrcnn_targets, mrcnn_outs)

    return rpn_loss + mrcnn_loss

import torch
import torch.nn.functional as F
import mrcnn.config


class Losses():
    def __init__(self, rpn_class=0.0, rpn_bbox=0.0, mrcnn_class=0.0,
                 mrcnn_bbox=0.0, mrcnn_mask=0.0):
        self.rpn_class = rpn_class
        self.rpn_bbox = rpn_bbox
        self.mrcnn_class = mrcnn_class
        self.mrcnn_bbox = mrcnn_bbox
        self.mrcnn_mask = mrcnn_mask
        self.update_total_loss()

    def to_item(self):
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


############################################################
#  Loss Functions
############################################################


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
    indices = torch.nonzero(rpn_match != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.detach()[:, 0],
                                        indices.detach()[:, 1], :]
    anchor_class = anchor_class[indices.detach()[:, 0], indices.detach()[:, 1]]

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
    indices = torch.nonzero(rpn_match == 1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.detach()[:, 0], indices.detach()[:, 1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0, :rpn_bbox.size()[0], :]

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
        loss = torch.tensor([0], dtype=torch.float32,
                            device=mrcnn.config.DEVICE)

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
        loss = torch.tensor([0], dtype=torch.float32,
                            device=mrcnn.config.DEVICE)

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
        loss = torch.tensor([0], dtype=torch.float32,
                            device=mrcnn.config.DEVICE)

    return loss


def compute_iou_loss(gt_masks, gt_boxes, gt_class_ids, mrcnn_masks,
                     mrcnn_boxes, mrcnn_class_logits, image_metas,
                     rois, config, mrcnn_probs):
    """Compute loss for single image according to:
       https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    """
    print(f"mrcnn_masks {mrcnn_masks.shape}")  # X, 28, 28
    # print("mrcnn_boxes {}".format(mrcnn_boxes.size()))  # X, 2, 4
    # print("mrcnn_class_logits {}".format(mrcnn_class_logits.size()))  # X, 2
    # print(f"mrcnn_probs {mrcnn_probs.size()}")  # X, 2
    # print("gt_boxes {}".format(gt_boxes.size()))  # 200, 4
    # print("gt_class_ids {}".format(gt_class_ids.size()))  # 200
    # print("rois {}".format(rois.size()))  # 200

    # print("image_metas {}".format(image_metas))
    # id, h, w, d, xi, yi, xf, yf, [active_class_ids]

    if mrcnn_class_logits.nelement() == 0:
        print("no predictions")
        return

    # 1 - unmold gt and mrcnn masks

    # remove zeros (padding)
    image_shape = np.array(image_metas[1:4])
    window = np.array(image_metas[4:8])
    # print(image_shape)
    # print(gt_class_ids.shape)
    zero_ix = np.where(gt_class_ids == 0)[0]
    # print(f"{zero_ix.shape}")
    N = zero_ix[0] if zero_ix.shape[0] > 0 else gt_class_ids.shape[0]

    # to numpy
    detections = detection_layer2(config, rois, mrcnn_probs, mrcnn_boxes, image_metas.unsqueeze(0))

    #detections = detections.detach().cpu().numpy()
    mrcnn_masks = mrcnn_masks.permute(0, 2, 3, 1)#.detach().cpu().numpy()
    _, _, _, final_masks =\
        unmold_detections2(detections, mrcnn_masks,
                           image_shape[:2], window)
    # print(f"final_rois {final_rois.shape}")
    # print(f"final_class_ids {final_class_ids.shape}")
    # print(f"final_masks {final_masks.shape}")
    # print(f"detections {detections.shape}")
    # print(f"ids {ids.size()}")
    # print(f"mrcnn_masks {mrcnn_masks.shape}")

    gt_masks = gt_masks[:N].detach()
    gt_boxes = gt_boxes[:N].detach()
    print(f"gt_masks {gt_masks.shape}")  # 200, 56, 56
    print(f"gt_boxes {gt_boxes.shape}")  # 200, 56, 56

    # print(f"masks {gt_masks.shape}")
    _, _, full_gt_masks =\
        unmold_boxes(gt_boxes, gt_class_ids, gt_masks,
                     image_shape[:2], window)
    #print(gt_masks.shape)
    #print(gt_boxes.shape)
    #print(gt_class_ids.shape)
    #print(image_shape[:2])

    # Compute scale and shift to translate coordinates to image domain.
    #``h_scale = image_shape[0] / (window[2] - window[0])
    #``w_scale = image_shape[1] / (window[3] - window[1])
    #``# scale = min(h_scale, w_scale)
    #``shift = window[:2]  # y, x
    #``scales = np.array([h_scale, w_scale, h_scale, w_scale])
    #``shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

    #``# Translate bounding boxes to image domain
    #``gt_boxes = np.multiply(gt_boxes - shifts, scales).astype(np.int32)

    # move gt masks to image domain
    #full_gt_masks = []
    #for i in range(N):
    #    # Convert neural network mask to full size mask
    #    # print(gt_boxes[i])
    #    y1, x1, y2, x2 = gt_boxes[i]
    #    # TODO: gt_masks by default should not have problems
    #    # use original masks
    #    if y2-y1 <= 0 or x2-x1 <= 0:
    #        continue
    #    full_mask = utils.unmold_mask(gt_masks[i], gt_boxes[i].astype(np.int),
    #                                  image_shape)
    #    full_gt_masks.append(full_mask)
    #full_gt_masks = np.stack(full_gt_masks, axis=-1)\
    #    if full_gt_masks else np.empty((0,) + gt_masks.shape[1:3])

    # print("gt_masks in image domain {}".format(full_gt_masks.shape))

    # compute IOUs
    full_gt_masks = full_gt_masks.to(torch.uint8)
    final_masks = final_masks.to(torch.uint8)
    print(f"final_gt_masks {full_gt_masks.shape}")
    print(f"final_masks {final_masks.shape}")
    ious = torch.zeros((full_gt_masks.shape[2], final_masks.shape[2]),
                       dtype=torch.float)
    print(f"{full_gt_masks.shape[2]} x {final_masks.shape[2]}")
    for gt_idx in range(0, full_gt_masks.shape[2]):
        for pred_idx in range(0, final_masks.shape[2]):
            # intersection = np.logical_and(final_masks[:, :, pred_idx],
            #                              full_gt_masks[:, :, gt_idx])
            intersection = final_masks[:, :, pred_idx] & full_gt_masks[:, :, gt_idx]
            # intersection = np.count_nonzero(intersection)
            intersection = torch.nonzero(intersection).shape[0]
            # union = np.logical_or(final_masks[:, :, pred_idx],
            #                      full_gt_masks[:, :, gt_idx])
            # union = np.count_nonzero(union)
            union = final_masks[:, :, pred_idx] | full_gt_masks[:, :, gt_idx]
            union = torch.nonzero(union).shape[0]
            iou = intersection/union if union != 0.0 else 0.0
            # if union == 0:
            #     print(f"{gt_idx} {pred_idx} {intersection} {union}")
            #     gt_area = torch.nonzero(full_gt_masks[:, :, gt_idx]).shape[0]
            #     pred_area = torch.nonzero(final_masks[:, :, pred_idx]).shape[0]
            #     print(f"{gt_area} {pred_area}")
            ious[gt_idx, pred_idx] = iou
            # print(f"{gt_idx} {pred_idx} {intersection} {union} {iou}")

    # compute hits
    thresholds = torch.arange(0.5, 1.0, 0.05)
    precisions = torch.empty_like(thresholds, device=mrcnn.config.DEVICE)
    for thresh_idx, threshold in enumerate(thresholds):
        hits = ious > threshold
        tp = torch.nonzero(hits.sum(dim=0)).shape[0]
        fp = torch.nonzero(hits.sum(dim=1) == 0).shape[0]
        fn = torch.nonzero(hits.sum(dim=0) == 0).shape[0]
        precisions[thresh_idx] = tp/(tp + fp + fn)
        # print(f"{precision}")

    # average precisions
    precision = precisions.mean()
    print(f"precision: {precision}")

    return


def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox,
                   target_class_ids, mrcnn_class_logits, target_deltas,
                   mrcnn_bbox, target_mask, mrcnn_mask):

    # print(rpn_match[0].size()) # 65472, 1, const
    # print(rpn_bbox[0].size()) # 64, 4, const
    # print(rpn_class_logits[0].size()) # 65472, 2, const
    # print(rpn_pred_bbox[0].size()) # 65472, 4, const
    # print(target_class_ids[0].size()) # 3/109
    # print(target_deltas[0].size()) # 3/109, 4
    # print(mrcnn_bbox[0].size()) # 3/109, 2, 4
    # print(target_mask[0].size()) # 3/109, 28, 28
    # print(mrcnn_mask[0].size()) # 3/109, 2, 28, 28
    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = torch.tensor([0.0], dtype=torch.float32,
                                    device=mrcnn.config.DEVICE)
    mrcnn_bbox_loss = torch.tensor([0.0], dtype=torch.float32,
                                   device=mrcnn.config.DEVICE)
    mrcnn_mask_loss = torch.tensor([0.0], dtype=torch.float32,
                                   device=mrcnn.config.DEVICE)
    for batch in range(0, len(target_class_ids)):
        mrcnn_class_loss += compute_mrcnn_class_loss(target_class_ids[batch],
                                                     mrcnn_class_logits[batch])
        mrcnn_bbox_loss += compute_mrcnn_bbox_loss(target_deltas[batch],
                                                   target_class_ids[batch],
                                                   mrcnn_bbox[batch])
        mrcnn_mask_loss += compute_mrcnn_mask_loss(target_mask[batch],
                                                   target_class_ids[batch],
                                                   mrcnn_mask[batch])

    if len(mrcnn_class_logits) != 0:
        mrcnn_class_loss /= len(target_class_ids)
    if len(mrcnn_bbox) != 0:
        mrcnn_bbox_loss /= len(target_class_ids)
    if len(mrcnn_mask) != 0:
        mrcnn_mask_loss /= len(target_class_ids)

    return Losses(rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss,
                  mrcnn_bbox_loss, mrcnn_mask_loss)



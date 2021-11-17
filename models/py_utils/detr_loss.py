import torch
import torch.nn.functional as F
from torch import nn

from .misc import (NestedTensor, nested_tensor_from_tensor_list,
                   accuracy, get_world_size, interpolate,
                   is_dist_avail_and_initialized)


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # threshold = 15 / 720.
        # self.threshold = nn.Threshold(threshold**2, 0.)
        # empty_weight = torch.ones(self.num_classes + 1)
        empty_weight = torch.ones(self.num_classes)
        # empty_weight[-1] = self.eos_coef

        self.register_buffer('empty_weight', empty_weight)
        print(self.empty_weight)

    def loss_labels(self, outputs, targets, indices, num_curves, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([tgt[:, 0][J].long() for tgt, (_, J) in zip (targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_curves):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([tgt.shape[0] for tgt in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_curves(self, outputs, targets, indices, num_curves):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_curves' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_lowers = outputs['pred_curves'][:, :, 0][idx]
        src_uppers = outputs['pred_curves'][:, :, 1][idx]
        src_polys  = outputs['pred_curves'][:, :, 2:][idx]
        target_lowers = torch.cat([tgt[:, 1][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_uppers = torch.cat([tgt[:, 2][i] for tgt, (_, i) in zip(targets, indices)], dim=0)
        target_points = torch.cat([tgt[:, 3:][i] for tgt, (_, i) in zip(targets, indices)], dim=0)

        target_xs = target_points[:, :target_points.shape[1] // 2]
        ys = target_points[:, target_points.shape[1] // 2:].transpose(1, 0)
        valid_xs = target_xs >= 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32)) ** 0.5
        weights = weights / torch.max(weights)

        # Calculate the predicted xs
        pred_xs = src_polys[:, 0] / (ys - src_polys[:, 1]) ** 2 + src_polys[:, 2] / (ys - src_polys[:, 1]) + \
                  src_polys[:, 3] + src_polys[:, 4] * ys - src_polys[:, 5]

        pred_xs = pred_xs * weights
        pred_xs = pred_xs.transpose(1, 0)
        target_xs = target_xs.transpose(1, 0) * weights
        target_xs = target_xs.transpose(1, 0)

        loss_lowers = F.l1_loss(src_lowers, target_lowers, reduction='none')
        loss_uppers = F.l1_loss(src_uppers, target_uppers, reduction='none')
        loss_polys  = F.l1_loss(pred_xs[valid_xs], target_xs[valid_xs], reduction='none')

        losses = {}
        losses['loss_lowers']  = loss_lowers.sum() / num_curves
        losses['loss_uppers']  = loss_uppers.sum() / num_curves
        losses['loss_curves']   = loss_polys.sum() / num_curves

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_curves, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'curves': self.loss_curves,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_curves, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_curves = sum(tgt.shape[0] for tgt in targets)
        num_curves = torch.as_tensor([num_curves], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_curves)
        num_curves = torch.clamp(num_curves / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_curves))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_curves, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, indices

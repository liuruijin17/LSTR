import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_encoding import build_position_encoding
from .transformer import build_transformer
from .detr_loss import SetCriterion
from .matcher import build_matcher

from .misc import *

from sample.vis import save_debug_images_boxes

BN_MOMENTUM = 0.1

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class kp(nn.Module):
    def __init__(self,
                 flag=False,
                 block=None,
                 layers=None,
                 res_dims=None,
                 res_strides=None,
                 attn_dim=None,
                 num_queries=None,
                 aux_loss=None,
                 pos_type=None,
                 drop_out=0.1,
                 num_heads=None,
                 dim_feedforward=None,
                 enc_layers=None,
                 dec_layers=None,
                 pre_norm=None,
                 return_intermediate=None,
                 lsp_dim=None,
                 mlp_layers=None,
                 num_cls=None,
                 norm_layer=FrozenBatchNorm2d
                 ):
        super(kp, self).__init__()
        self.flag = flag
        # above all waste not used
        self.norm_layer = norm_layer

        self.inplanes = res_dims[0]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, res_dims[0], layers[0], stride=res_strides[0])
        self.layer2 = self._make_layer(block, res_dims[1], layers[1], stride=res_strides[1])
        self.layer3 = self._make_layer(block, res_dims[2], layers[2], stride=res_strides[2])
        self.layer4 = self._make_layer(block, res_dims[3], layers[3], stride=res_strides[3])

        hidden_dim = attn_dim
        self.aux_loss = aux_loss
        self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, type=pos_type)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(res_dims[-1], hidden_dim, kernel_size=1)  # the same as channel of self.layer4

        self.transformer = build_transformer(hidden_dim=hidden_dim,
                                             dropout=drop_out,
                                             nheads=num_heads,
                                             dim_feedforward=dim_feedforward,
                                             enc_layers=enc_layers,
                                             dec_layers=dec_layers,
                                             pre_norm=pre_norm,
                                             return_intermediate_dec=return_intermediate)

        self.class_embed    = nn.Linear(hidden_dim, num_cls + 1)
        self.specific_embed = MLP(hidden_dim, hidden_dim, lsp_dim - 4, mlp_layers)
        self.shared_embed   = MLP(hidden_dim, hidden_dim, 4, mlp_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _train(self, *xs, **kwargs):
        images = xs[0]  # B 3 360 640
        masks  = xs[1]  # B 1 360 640

        p = self.conv1(images)  # B 16 180 320
        p = self.bn1(p)  # B 16 180 320
        p = self.relu(p)  # B 16 180 320
        p = self.maxpool(p)  # B 16 90 160
        p = self.layer1(p)  # B 16 90 160
        p = self.layer2(p)  # B 32 45 80
        p = self.layer3(p)  # B 64 23 40
        p = self.layer4(p)  # B 128 12 20
        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos    = self.position_embedding(p, pmasks)
        hs, _, weights  = self.transformer(self.input_proj(p), pmasks, self.query_embed.weight, pos)
        output_class    = self.class_embed(hs)
        output_specific = self.specific_embed(hs)
        output_shared   = self.shared_embed(hs)
        output_shared   = torch.mean(output_shared, dim=-2, keepdim=True)
        output_shared   = output_shared.repeat(1, 1, output_specific.shape[2], 1)
        output_specific = torch.cat([output_specific[:, :, :, :2], output_shared, output_specific[:, :, :, 2:]], dim=-1)
        out = {'pred_logits': output_class[-1], 'pred_curves': output_specific[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_class, output_specific)
        return out, weights

    def _test(self, *xs, **kwargs):
        return self._train(*xs, **kwargs)

    def forward(self, *xs, **kwargs):
        if self.flag:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_curves': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class AELoss(nn.Module):
    def __init__(self,
                 debug_path=None,
                 aux_loss=None,
                 num_classes=None,
                 dec_layers=None
                 ):
        super(AELoss, self).__init__()
        self.debug_path  = debug_path
        weight_dict = {'loss_ce': 3, 'loss_curves': 5, 'loss_lowers': 2, 'loss_uppers': 2}
        # cardinality is not used to propagate loss
        matcher = build_matcher(set_cost_class=weight_dict['loss_ce'],
                                curves_weight=weight_dict['loss_curves'],
                                lower_weight=weight_dict['loss_lowers'],
                                upper_weight=weight_dict['loss_uppers'])
        losses  = ['labels', 'curves', 'cardinality']

        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(num_classes=num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=1.0,
                                      losses=losses)

    def forward(self,
                iteration,
                save,
                viz_split,
                outputs,
                targets):

        gt_cluxy = [tgt[0] for tgt in targets[1:]]
        loss_dict, indices = self.criterion(outputs, gt_cluxy)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Save detected images during training
        if save:
            which_stack = 0
            save_dir = os.path.join(self.debug_path, viz_split)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_name = 'iter_{}_layer_{}'.format(iteration % 5000, which_stack)
            save_path = os.path.join(save_dir, save_name)
            with torch.no_grad():
                gt_viz_inputs = targets[0]
                tgt_labels = [tgt[:, 0].long() for tgt in gt_cluxy]
                pred_labels = outputs['pred_logits'].detach()
                prob = F.softmax(pred_labels, -1)
                scores, pred_labels = prob.max(-1)  # 4 10

                pred_curves = outputs['pred_curves'].detach()
                pred_clua3a2a1a0 = torch.cat([scores.unsqueeze(-1), pred_curves], dim=-1)

                save_debug_images_boxes(gt_viz_inputs,
                                        tgt_curves=gt_cluxy,
                                        tgt_labels=tgt_labels,
                                        pred_curves=pred_clua3a2a1a0,
                                        pred_labels=pred_labels,
                                        prefix=save_path)

        return (losses, loss_dict_reduced, loss_dict_reduced_unscaled,
                loss_dict_reduced_scaled, loss_value)

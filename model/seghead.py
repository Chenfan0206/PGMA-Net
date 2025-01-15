from typing import Optional
from typing import Union
from torch import Tensor
from torch.nn.functional import one_hot
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from mmcv.utils import print_log
import os
import sys
sys.path.append('repository/MaskCLIP/mmseg')
from mmseg.utils import get_root_logger
from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from torch.autograd import Variable
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import cv2, random, string, os
import numpy as np
import einops
import math
import random
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
import functools
import mmcv
import torch.nn.functional as F


voc_classes = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling', 'tile ceiling', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk', 'dirt', 'door', 'fence', 'marble floor', 'floor', 'stone floor', 'tile floor', 'wood floor', 'flower', 'fog', 'food', 'fruit', 'furniture', 'grass', 'gravel', 'ground', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky', 'skyscraper', 'snow', 'solid', 'stairs', 'stone', 'straw', 'structural', 'table', 'tent', 'textile', 'towel', 'tree', 'vegetable', 'brick wall', 'concrete wall', 'wall', 'panel wall', 'stone wall', 'tile wall', 'wood wall', 'water', 'waterdrops', 'blind window', 'window', 'wood'][:80]

class_names = voc_classes





def get_class_weight(class_weight):
    if isinstance(class_weight, str):
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            class_weight = mmcv.load(class_weight)

    return class_weight


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
    return wrapper

def get_class_weight(class_weight):
    if isinstance(class_weight, str):
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            class_weight = mmcv.load(class_weight)
    return class_weight

@weighted_loss
def dice_loss(pred,
              target,
              valid_mask,
              smooth=1,
              exponent=2,
              class_weight=None,
              ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes


@weighted_loss
def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwards):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den


class DiceLoss(nn.Module):
    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_dice',
                 **kwards):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwards):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        return self._loss_name


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss



def py_sigmoid_focal_loss(pred,
                          target,
                          one_hot_target=None,
                          weight=None,
                          gamma=2.0,
                          alpha=0.5,
                          class_weight=None,
                          valid_mask=None,
                          reduction='mean',
                          avg_factor=None):
    if isinstance(alpha, list):
        alpha = pred.new_tensor(alpha)
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * one_minus_pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    final_weight = torch.ones(1, pred.size(1)).type_as(loss)
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       one_hot_target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.5,
                       class_weight=None,
                       valid_mask=None,
                       reduction='mean',
                       avg_factor=None):
    final_weight = torch.ones(1, pred.size(1)).type_as(pred)
    if isinstance(alpha, list):
        loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
                                   gamma, 0.5, None, 'none') * 2
        alpha = pred.new_tensor(alpha)
        final_weight = final_weight * (
            alpha * one_hot_target + (1 - alpha) * (1 - one_hot_target))
    else:
        loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
                                   gamma, alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss

class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.5,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_focal'):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, \
            'AssertionError: Only sigmoid focal loss supported now.'
        assert reduction in ('none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert isinstance(alpha, (float, list)), \
            'AssertionError: alpha should be of type float'
        assert isinstance(gamma, float), \
            'AssertionError: gamma should be of type float'
        assert isinstance(loss_weight, float), \
            'AssertionError: loss_weight should be of type float'
        assert isinstance(loss_name, str), \
            'AssertionError: loss_name should be of type str'
        assert isinstance(class_weight, list) or class_weight is None, \
            'AssertionError: class_weight must be None or of type list'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
            (pred.size(0) == target.size(0) and
             pred.shape[2:] == target.shape[1:]), \
            "The shape of pred doesn't match the shape of target"

        original_shape = pred.shape
        pred = pred.transpose(0, 1)
        pred = pred.reshape(pred.size(0), -1)
        pred = pred.transpose(0, 1).contiguous()

        if original_shape == target.shape:
            target = target.transpose(0, 1)
            target = target.reshape(target.size(0), -1)
            target = target.transpose(0, 1).contiguous()
        else:
            target = target.view(-1).contiguous()
            valid_mask = (target != ignore_index).view(-1, 1)
            target = torch.where(target == ignore_index, target.new_tensor(0),
                                 target)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.size(1)
            if torch.cuda.is_available() and pred.is_cuda:
                if target.dim() == 1:
                    one_hot_target = F.one_hot(target, num_classes=num_classes)
                else:
                    one_hot_target = target
                    target = target.argmax(dim=1)
                    valid_mask = (target != ignore_index).view(-1, 1)
                calculate_loss_func = sigmoid_focal_loss
            else:
                one_hot_target = None
                if target.dim() == 1:
                    target = F.one_hot(target, num_classes=num_classes)
                else:
                    valid_mask = (target.argmax(dim=1) != ignore_index).view(
                        -1, 1)
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                one_hot_target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                class_weight=self.class_weight,
                valid_mask=valid_mask,
                reduction=reduction,
                avg_factor=avg_factor)

            if reduction == 'none':
                loss_cls = loss_cls.transpose(0, 1)
                loss_cls = loss_cls.reshape(original_shape[1],
                                            original_shape[0],
                                            *original_shape[2:])
                loss_cls = loss_cls.transpose(0, 1).contiguous()
        else:
            raise NotImplementedError
        return loss_cls


fl = FocalLoss()


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    def forward(self, logits, targets):
        bs = targets.size(0)
        smooth = 1
        probs = F.sigmoid(logits)
        m1 = probs.view(bs, -1)
        m2 = targets.contiguous().view(bs, -1)
        intersection = m1 * m2
        score = 2.0 * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / bs
        return score


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)
        if kv_include_self:
            context = torch.cat(
                (x, context), dim=1
            )  
        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


@HEADS.register_module()
class seghead(BaseDecodeHead):
    def __init__(
        self,
        text_categories,
        text_channels,
        text_embeddings_path,
        visual_projs_path,
        vit=False,
        ks_thresh=0.0,
        pd_thresh=0.0,
        attn_pooling=False,
        num_heads=32,
        decode_mode='text_as_classifier',
        other_args=None,
        **kwargs,
    ):
        super(seghead, self).__init__(**kwargs)

        self.text_categories = text_categories
        self.text_channels = text_channels
        self.text_embeddings_path = text_embeddings_path
        self.visual_projs_path = visual_projs_path
        self.other_args = other_args

        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(
                torch.zeros(text_categories, text_channels)
            )
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            print('Loading text embeddings from {}'.format(self.text_embeddings_path))
            self.register_buffer(
                'text_embeddings', torch.randn(text_categories, text_channels)
            )
            self.load_text_embeddings()

        self.vit = vit
        if vit:
            self.proj = nn.Conv2d(self.in_channels, text_channels, 1, bias=False)
        else:
            self.q_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.k_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.v_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.c_proj = nn.Conv2d(self.in_channels, text_channels, 1)
        self.load_visual_projs()

        self.ks_thresh = ks_thresh
        self.pd_thresh = pd_thresh
        self.attn_pooling = attn_pooling
        self.num_heads = num_heads

        self.decode_mode = decode_mode

        if False:
            pass
        
        elif self.decode_mode.startswith('clip_context_affinity'):
            if self.decode_mode == "clip_context_affinity_v17":
                in_channels = [256, 512, 1024, 2048]
                self.nlayers = self.other_args['nlayers']
                self.combinations = ['clip', 'context', 'Aqq@clip', 'Aqq@context',
                                     'Asq@Ms', 'Asq@Ms_clip', "contextfake", "Aqq@contextfake", "Asqfake@clip"]
                outch1, outch2, outch3 = 16, 64, 128
                self.conv_stage4 = self.build_conv_block(
                    len(self.combinations)*self.nlayers[-1],
                    [outch1, outch2, outch3],
                    [3, 3, 3],
                    [1, 1, 1],
                ) 
                self.conv_stage3 = self.build_conv_block(
                    len(self.combinations)*self.nlayers[-2],
                    [outch1, outch2, outch3],
                    [5, 3, 3],
                    [1, 1, 1],
                ) 
                self.conv4_3 = self.build_conv_block(
                    2*outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]
                ) 

                self.mixer1 = nn.Sequential(
                    nn.Conv2d(
                        outch3+in_channels[1]*1,
                        outch3,
                        (3, 3),
                        padding=(1, 1),
                        bias=True,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(outch3, outch2, (3, 3),
                              padding=(1, 1), bias=True),
                    nn.ReLU(),
                )

                self.mixer2 = nn.Sequential(
                    nn.Conv2d(outch2 + in_channels[0]*1, outch2, (3, 3),
                              padding=(1, 1), bias=True),
                    nn.ReLU(),
                    nn.Conv2d(outch2, outch1, (3, 3),
                              padding=(1, 1), bias=True),
                    nn.ReLU(),
                )

                self.mixer3 = nn.Sequential(
                    nn.Conv2d(outch1, outch1, (3, 3),
                              padding=(1, 1), bias=True),
                    nn.ReLU(),
                    nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True),
                )

                self.affinity_blocks = nn.ModuleList()
                for grid_n in [12**2]:
                    self.affinity_blocks.append(CrossAttention(
                        dim=grid_n, heads=8, dim_head=grid_n, dropout=0.1))

                for grid_n in [12**2]:
                    self.affinity_blocks.append(CrossAttention(
                        dim=grid_n, heads=8, dim_head=grid_n, dropout=0.1))

                need_init_modules = [self.conv_stage4, self.conv_stage3,
                                     self.conv4_3,  self.mixer1, self.mixer2, self.mixer3, self.affinity_blocks]
                for module in need_init_modules:
                    for m in module.modules():
                        if isinstance(m, nn.Conv2d):
                            nn.init.kaiming_normal_(
                                m.weight, mode='fan_out', nonlinearity='relu')
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.BatchNorm2d):
                            nn.init.constant_(m.weight, 1)
                            nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.GroupNorm):
                            nn.init.constant_(m.weight, 1)
                            nn.init.constant_(m.bias, 0)
                        elif isinstance(m, nn.Linear):
                            nn.init.normal_(m.weight, 0, 0.01)
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0)



    def init_weights(self):
        if self.text_embeddings_path is None:
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.load_text_embeddings()
        self.load_visual_projs()

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cuda')
        if loaded.shape[0] == 171:
           loaded = loaded[:80, :]
        self.text_embeddings[:, :] = loaded[:, :]
        print_log(
            f'Loaded text embeddings from {self.text_embeddings_path}',
            logger=get_root_logger(),
        )

    def load_visual_projs(self):
        loaded = torch.load(self.visual_projs_path, map_location='cuda')
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(
            f'Loaded proj weights from {self.visual_projs_path}',
            logger=get_root_logger(),
        )

    def get_feat_from_inputs(self, inputs):
        hw_shape = inputs[-1][-1].shape[-2:]
        x = inputs[-1]

        q, k, v, cls_token = None, None, None, None
        if self.vit:
            if isinstance(x, list) and len(x) == 4:
                x, q, k, v = x
            if isinstance(x, list) and len(x) == 2:
                x, cls_token = x
            if v is not None:
                feat = self.proj(v)
            else:
                feat = self.proj(x)
            if cls_token is not None:
                cls_token = self.proj(cls_token[:, :, None, None])[:, :, 0, 0]
        else:
            if self.attn_pooling:
                N, C, H, W = x.shape
                x = x.view(N, C, -1).permute(2, 0, 1) 
                x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
                x, _ = F.multi_head_attention_forward(
                    query=x,
                    key=x,
                    value=x,
                    embed_dim_to_check=x.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight[:, :, 0, 0],
                    k_proj_weight=self.k_proj.weight[:, :, 0, 0],
                    v_proj_weight=self.v_proj.weight[:, :, 0, 0],
                    in_proj_weight=None,
                    in_proj_bias=torch.cat(
                        [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
                    ),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight[:, :, 0, 0],
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False,
                )
                feat = x[1:].permute(1, 2, 0).view(N, -1, H, W)
            else:
                q = self.q_proj(x)
                k = self.k_proj(x)
                q = torch.flatten(q, start_dim=2).transpose(-2, -1)
                k = torch.flatten(k, start_dim=2).transpose(-2, -1)
                v = self.v_proj(x)
                feat = self.c_proj(v)

        return (x, q, k, v, cls_token, feat, hw_shape)

    def forward_original_clip_text_as_classifier(self, batch,_forwad_type='query'):
        inputs = batch['query_feat'] if _forwad_type == 'query' else batch['support_feat']
        _, _, _, _, _, feat, hw_shape = self.get_feat_from_inputs(inputs)
        output = self.cls_seg(feat)
        _size  = batch['query_mask'].shape[-2:] if _forwad_type == 'query' else batch['support_masks'].shape[-2:]
        output = resize(input=output,size=_size,mode='bilinear',align_corners=False)
        pred_mask_c = output.argmax(dim=1)
        pred_mask_01 = torch.zeros_like(pred_mask_c)
        pred_mask_01[pred_mask_c == batch['class_id'].unsqueeze(-1).unsqueeze(-1)] = 1
        tmp = torch.nn.functional.softmax(output, dim=1)
        logits_by_softmax = tmp[
            torch.arange(tmp.shape[0]), batch['class_id']
        ].unsqueeze(1)
        _min = einops.reduce(logits_by_softmax, 'n c h w -> n c () ()', 'min')
        _max = einops.reduce(logits_by_softmax, 'n c h w -> n c () ()', 'max')
        logits_by_softmax = (logits_by_softmax - _min) / (_max - _min + 1e-8)
        _all = {
            'pred_logits': output,
            'pred_mask_01': pred_mask_01,
            'pred_mask_c': pred_mask_c,
            "logits_by_softmax": logits_by_softmax
        }
        return _all

    def forward(self, batch):
        inputs = batch['query_feat']
        hw_shape = batch['query_feat'][-1][-1].shape[-2:]
        x = inputs[-1]
        q, k, v, cls_token = None, None, None, None
        if self.vit:
            if isinstance(x, list) and len(x) == 4:
                x, q, k, v = x
            if isinstance(x, list) and len(x) == 2:
                x, cls_token = x
            if v is not None:
                feat = self.proj(v)
            else:
                feat = self.proj(x)
            if cls_token is not None:
                cls_token = self.proj(cls_token[:, :, None, None])[:, :, 0, 0]
        else:
            if self.attn_pooling:
                N, C, H, W = x.shape
                x = x.view(N, C, -1).permute(2, 0, 1)
                x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
                x, _ = F.multi_head_attention_forward(
                    query=x,
                    key=x,
                    value=x,
                    embed_dim_to_check=x.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight[:, :, 0, 0],
                    k_proj_weight=self.k_proj.weight[:, :, 0, 0],
                    v_proj_weight=self.v_proj.weight[:, :, 0, 0],
                    in_proj_weight=None,
                    in_proj_bias=torch.cat(
                        [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
                    ),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight[:, :, 0, 0],
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False,
                )
                feat = x[1:].permute(1, 2, 0).view(N, -1, H, W)
            else:
                q = self.q_proj(x)
                k = self.k_proj(x)
                q = torch.flatten(q, start_dim=2).transpose(-2, -1)
                k = torch.flatten(k, start_dim=2).transpose(-2, -1)
                v = self.v_proj(x)
                feat = self.c_proj(v)

        if self.decode_mode == 'clip_context_affinity_v17':
            coarse_masks = []
            with torch.no_grad():
                for context_prob, A, contextfake, Asqfake in zip(batch['context_probs'], batch['affinity'], batch['contextfake'], batch['Asqfake']):
                    coarse_masks.append(self.get_clip_guided_mask_withdropout(infos=(batch['clip_probs'], context_prob.unsqueeze(
                        1), A, batch['support_masks'], batch['support_clip_probs'], contextfake.unsqueeze(
                        1), Asqfake), combinations=self.combinations))
            

            coarse_masks_affinity = []
            for _idx, (context_prob, A, contextfake, Asqfake) in enumerate(zip(batch['context_probs'], batch['affinity'], batch['contextfake'], batch['Asqfake'])):
                Ass, Asq, Aqq = A
                if Ass.shape[-1]==144:
                    _Asq = self.affinity_blocks[0](x=Ass,context=einops.rearrange(Asq,'b hsws hqwq -> b hqwq hsws'))
                    _Asq = einops.rearrange(_Asq, 'b hqwq hsws -> b hsws hqwq')
                    _Asq = torch.sigmoid(_Asq)
                    _Aqq = self.affinity_blocks[1](x=_Asq,context=Aqq) 
                    _Aqq = torch.sigmoid(_Aqq)
                    A = (Ass,_Asq,_Aqq)
                coarse_masks_affinity.append(self.get_clip_guided_mask_withdropout(infos=(batch['clip_probs'], context_prob.unsqueeze(
                    1), A, batch['support_masks'], batch['support_clip_probs'], contextfake.unsqueeze(
                    1), Asqfake), combinations=self.combinations))

            p_mask_with_query_without_affinity = torch.tensor(
                data=[1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=coarse_masks[0].dtype, device=coarse_masks[0].device)
            p_mask_with_query_without_affinity = p_mask_with_query_without_affinity.view(1,p_mask_with_query_without_affinity.shape[0],1,1)

            p_mask_with_query_with_affinity = torch.tensor(
                data=[1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=coarse_masks[0].dtype, device=coarse_masks[0].device)
            p_mask_with_query_with_affinity = p_mask_with_query_with_affinity.view(1,p_mask_with_query_with_affinity.shape[0],1,1)

            p_mask_with_sq_without_affinity = torch.tensor(
                data=[1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=coarse_masks[0].dtype, device=coarse_masks[0].device)
            p_mask_with_sq_without_affinity = p_mask_with_sq_without_affinity.view(1,p_mask_with_sq_without_affinity.shape[0],1,1)

            p_mask_with_sq_with_affinity = torch.tensor(
                data=[1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=coarse_masks[0].dtype, device=coarse_masks[0].device)
            p_mask_with_sq_with_affinity = p_mask_with_sq_with_affinity.view(1,p_mask_with_sq_with_affinity.shape[0],1,1)
            p_mask_with_sq_with_affinity_without_sgt = torch.tensor(
                data=[1,0,1,0,0,0,1,1,1], dtype=coarse_masks[0].dtype, device=coarse_masks[0].device)
            p_mask_with_sq_with_affinity_without_sgt = p_mask_with_sq_with_affinity_without_sgt.view(1,p_mask_with_sq_with_affinity_without_sgt.shape[0],1,1)
            p_mask_all = torch.tensor(
                data=[1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=coarse_masks[0].dtype, device=coarse_masks[0].device)
            p_mask_all = p_mask_all.view(1,p_mask_all.shape[0],1,1)
            random_01_list = [0]
            while sum(random_01_list) == 0:
                random_01_list = [random.randint(0, 1) for i in range(9)]
                mask_random = torch.tensor(
                    data=random_01_list, dtype=coarse_masks[0].dtype, device=coarse_masks[0].device)
                mask_random = mask_random.view(1, mask_random.shape[0], 1, 1)
            only_training_free_affinity_weight = 1
            only_high_order_affinity_weight = 0
            affinity_fusion_weight = 0.5
            ablation_row_1 = [p_mask_with_query_without_affinity, only_training_free_affinity_weight]
            ablation_row_2 = [p_mask_with_query_with_affinity, only_training_free_affinity_weight]
            ablation_row_3 = [p_mask_with_query_with_affinity, only_high_order_affinity_weight]
            ablation_row_4 = [p_mask_with_query_with_affinity, affinity_fusion_weight]
            ablation_row_5 = [p_mask_with_sq_without_affinity, only_training_free_affinity_weight]
            ablation_row_6 = [p_mask_with_sq_with_affinity, only_training_free_affinity_weight]
            ablation_row_7 = [p_mask_with_sq_with_affinity, only_high_order_affinity_weight]
            ablation_row_8 = [p_mask_with_sq_with_affinity, affinity_fusion_weight]
            ablation_row_all = [p_mask_all, affinity_fusion_weight] 
            ablation_row_co_seg = [p_mask_with_sq_with_affinity_without_sgt, affinity_fusion_weight]
            ablation_random = [mask_random, random.choice([only_training_free_affinity_weight,only_high_order_affinity_weight,affinity_fusion_weight])]
            # training_state =  random.choices([ablation_row_8,ablation_row_co_seg,ablation_row_1,ablation_row_2,ablation_row_3,ablation_row_4,ablation_row_5,ablation_row_6,ablation_row_7,ablation_random], [0.4,0.2,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05])[0]
            # training_state =  random.choices([ablation_row_all,ablation_row_4,ablation_row_co_seg,ablation_row_1,ablation_row_2,ablation_row_3,ablation_row_8,ablation_row_5,ablation_row_6,ablation_row_7,ablation_random], [0.5,0.13,0.13,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03])[0]
            training_state =  random.choices([ablation_row_all,ablation_row_4,ablation_row_co_seg,ablation_row_1,ablation_row_2,ablation_row_3,ablation_row_8,ablation_row_5,ablation_row_6,ablation_row_7,ablation_random], [0.5,0.13,0.13,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03])[0]
            validation_state = ablation_row_all
            
            if self.other_args['use_dropout']:
                if self.training:
                    selected_mask =training_state[0]
                    training_free_affinity_weight = training_state[1]
                else:
                    selected_mask = validation_state[0]
                    training_free_affinity_weight = validation_state[1]
            else:
                selected_mask = validation_state[0]
                training_free_affinity_weight = validation_state[1]

            coarse_masks4 = coarse_masks[-self.nlayers[-1]:]
            _len = len(coarse_masks4)
            coarse_masks4 = einops.rearrange(coarse_masks4, 'l b c h w -> b (l c) h w')
            coarse_masks4_enhanced = coarse_masks_affinity[-self.nlayers[-1]:]
            coarse_masks4_enhanced = einops.rearrange( coarse_masks4_enhanced, 'l b c h w -> b (l c) h w')
            coarse_masks4 = coarse_masks4*training_free_affinity_weight + coarse_masks4_enhanced*(1-training_free_affinity_weight) 
            coarse_masks4 = coarse_masks4*einops.repeat(selected_mask, 'b c h w -> b (l c) h w', l=_len)
            coarse_masks4 = self.conv_stage4(coarse_masks4)
            coarse_masks3 = coarse_masks[-(self.nlayers[-1] + self.nlayers[-2]):-self.nlayers[-1]]
            _len = len(coarse_masks3)
            coarse_masks3 = einops.rearrange(coarse_masks3, 'l b c h w -> b (l c) h w')
            coarse_masks3 = coarse_masks3*einops.repeat(selected_mask, 'b c h w -> b (l c) h w', l=_len) 
            coarse_masks3 = self.conv_stage3(coarse_masks3)
            coarse_masks4 = F.interpolate(
                coarse_masks4,
                coarse_masks3.size()[-2:],
                mode='bilinear',
                align_corners=True,
            )
            mix = torch.cat((coarse_masks4, coarse_masks3), 1)
            mix = F.interpolate(
                self.conv4_3(mix) ,scale_factor=2 , mode='bilinear', align_corners=True
            )
            out = torch.cat(
                (
                    mix,
                    F.interpolate(
                        input=batch['query_feat'][self.nlayers[0]+self.nlayers[1]-1],
                        size=(mix.shape[-2], mix.shape[-1]),
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                1,
            )
            out = self.mixer1(out)  
            out = F.interpolate(
                out, scale_factor=2, mode='bilinear', align_corners=True
            )
            out = torch.cat(
                (out,
                batch['query_feat'][self.nlayers[0]-1]), 1)
            out = self.mixer2(out)  

            logit_mask = self.mixer3(out)  
            logit_mask = F.interpolate(
                logit_mask,
                batch['query_mask'].size()[-2:],
                mode='bilinear',
                align_corners=True,
            )

            fl_loss_weight = 0.5
            loss  = fl_loss_weight * fl(
                logit_mask, batch['query_mask'].long()
            ) + (1-fl_loss_weight) * DiceLoss()(logit_mask, batch['query_mask'].long())

            pred_mask_01 = torch.argmax(logit_mask, dim=1)
            _all = {
                'pred_logits': logit_mask,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all


    def min_max_norm(self, x):
        _max = einops.reduce(x, 'b c h w -> b c () ()', reduction='max')
        _min = einops.reduce(x, 'b c h w -> b c () ()', reduction='min')
        x = (x - _min) / (_max - _min + 1e-8)
        return x



    def get_clip_guided_mask(self,infos,combinations=['clip']):
        all_combination = ['clip', 'Aqq@clip', 'context','context+clip','Asq@Ms',"Asq@Ms+clip","Asq@Ms*clip"]
        for combination in combinations:
            assert combination in all_combination, "combination must be in {}".format(all_combination)
        clip_prob, context_prob, A,Ms_gt = infos
        Ass, Asq, Aqq = A
        _size = context_prob.shape[-1]
        clip_prob = F.interpolate(clip_prob, size=(_size,_size) , mode='bilinear', align_corners=True)        
        _all = []
        for combination in combinations:
            if combination == 'clip':
                _all.append(clip_prob)
            if combination == 'Aqq@clip':
                _clip = clip_prob.squeeze(1).view(clip_prob.shape[0], -1)            
                tmp = torch.einsum('bxy,by -> bx', torch.softmax(Aqq, dim=-1), _clip)
                _min = einops.reduce(
                    tmp, ' bsz hw -> bsz ()', reduction="min")
                _max = einops.reduce(
                    tmp, ' bsz hw -> bsz ()', reduction="max")
                tmp = einops.rearrange(
                    (tmp - _min) / (_max - _min + 1e-8), 'bsz (hq wq) -> bsz hq wq', hq=_size, wq=_size)
                _all.append(tmp.unsqueeze(1))
            if combination == 'context':
                _all.append(context_prob)
            if combination == 'context+clip':
                tmp = context_prob + clip_prob
                _min = einops.reduce(tmp, ' bsz c h w -> bsz c () ()', reduction="min")
                _max = einops.reduce(tmp, ' bsz c h w -> bsz c () ()', reduction="max")
                _all.append((tmp - _min) / (_max - _min + 1e-8))
            if combination == 'Asq@Ms': 
                ms = F.interpolate(input=Ms_gt, size=_size, mode='bilinear', align_corners=True).squeeze(1).view(clip_prob.shape[0], -1) 
                tmp = torch.einsum('bxy,bx -> by', torch.softmax(Asq, dim=-1), ms)
                _min = einops.reduce(
                    tmp, ' bsz hw -> bsz ()', reduction="min")
                _max = einops.reduce(
                    tmp, ' bsz hw -> bsz ()', reduction="max")
                tmp = einops.rearrange(
                    (tmp - _min) / (_max - _min + 1e-8), 'bsz (hq wq) -> bsz hq wq', hq=_size, wq=_size)
                _all.append(tmp.unsqueeze(1))
            if combination=='Asq@Ms+clip':
                ms = F.interpolate(input=Ms_gt, size=_size, mode='bilinear', align_corners=True).squeeze(1).view(clip_prob.shape[0], -1) 
                tmp = torch.einsum('bxy,bx -> by', torch.softmax(Asq, dim=-1), ms)
                _min = einops.reduce(
                    tmp, ' bsz hw -> bsz ()', reduction="min")
                _max = einops.reduce(
                    tmp, ' bsz hw -> bsz ()', reduction="max")
                tmp = einops.rearrange(
                    (tmp - _min) / (_max - _min + 1e-8), 'bsz (hq wq) -> bsz hq wq', hq=_size, wq=_size)
                tmp = tmp.unsqueeze(1)+clip_prob
                _min = einops.reduce(tmp, ' bsz c h w -> bsz c () ()', reduction="min")
                _max = einops.reduce(tmp, ' bsz c h w -> bsz c () ()', reduction="max")
                _all.append((tmp - _min) / (_max - _min + 1e-8))
            if combination == 'Asq@Ms*clip':
                ms = F.interpolate(input=Ms_gt, size=_size, mode='bilinear', align_corners=True).squeeze(
                    1).view(clip_prob.shape[0], -1)
                tmp = torch.einsum(
                    'bxy,bx -> by', torch.softmax(Asq, dim=-1), ms)
                _min = einops.reduce(
                    tmp, ' bsz hw -> bsz ()', reduction="min")
                _max = einops.reduce(
                    tmp, ' bsz hw -> bsz ()', reduction="max")
                tmp = einops.rearrange(
                    (tmp - _min) / (_max - _min + 1e-8), 'bsz (hq wq) -> bsz hq wq', hq=_size, wq=_size)
                tmp = tmp.unsqueeze(1)*clip_prob
                _min = einops.reduce(
                    tmp, ' bsz c h w -> bsz c () ()', reduction="min")
                _max = einops.reduce(
                    tmp, ' bsz c h w -> bsz c () ()', reduction="max")
                _all.append((tmp - _min) / (_max - _min + 1e-8))
        _all = einops.rearrange(_all, 'n bsz c h w -> bsz (n c) h w')
        return _all

    def get_clip_guided_mask_withdropout(self, infos, combinations=['clip']):
        all_combination = ['clip', 'context', 'Aqq@clip', "Aqq@context",'Asq@Ms', "Asq@Ms_clip"]
        all_combination.extend(["contextfake", "Aqq@contextfake", "Asqfake@clip"])
        for combination in combinations:
            assert combination in all_combination, "combination must be in {}".format(
                all_combination)

        clip_prob, context_prob, A, Ms_gt, support_clip_probs, contextfake,Asqfake= infos
        Ass, Asq, Aqq = A

        _size = context_prob.shape[-1]
        clip_prob = F.interpolate(clip_prob, size=(
            _size, _size), mode='bilinear', align_corners=True)
        _all = []
        
        empty = torch.zeros_like(clip_prob)
        for combination in all_combination:
            if combination not in combinations:
                _all.append(empty) 
            else:
                if combination == 'clip':
                    _all.append(clip_prob)
                if combination == 'Aqq@clip':
                    _clip = clip_prob.squeeze(1).view(clip_prob.shape[0], -1)
                    tmp = torch.einsum(
                        'bxy,by -> bx', torch.softmax(Aqq, dim=-1), _clip)
                    _min = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="min")
                    _max = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="max")
                    tmp = einops.rearrange(
                        (tmp - _min) / (_max - _min + 1e-8), 'bsz (hq wq) -> bsz hq wq', hq=_size, wq=_size)
                    _all.append(tmp.unsqueeze(1))
                if combination == 'context':
                    _all.append(context_prob)
                if combination == 'Asq@Ms':
                    ms = F.interpolate(input=Ms_gt, size=_size, mode='bilinear', align_corners=True).squeeze(
                        1).view(clip_prob.shape[0], -1)
                    tmp = torch.einsum(
                        'bxy,bx -> by', torch.softmax(Asq, dim=1), ms)
                    _min = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="min")
                    _max = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="max")
                    tmp = einops.rearrange(
                        (tmp - _min) / (_max - _min + 1e-8), 'bsz (hq wq) -> bsz hq wq', hq=_size, wq=_size)
                    _all.append(tmp.unsqueeze(1))

                if combination == "Aqq@context":
                    _context = context_prob.squeeze(1).view(context_prob.shape[0], -1)
                    tmp = torch.einsum(
                        'bxy,by -> bx', torch.softmax(Aqq, dim=-1), _context)
                    _min = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="min")
                    _max = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="max")
                    tmp = einops.rearrange(
                        (tmp - _min) / (_max - _min + 1e-8), 'bsz (hq wq) -> bsz hq wq', hq=_size, wq=_size)
                    _all.append(tmp.unsqueeze(1))

                if combination == "Asq@Ms_clip":
                    support_clip_probs = F.interpolate(support_clip_probs, size=(_size, _size), mode='bilinear', align_corners=True)
                    support_clip_probs = support_clip_probs.view(support_clip_probs.shape[0], -1)
                    tmp = torch.einsum('bxy,bx -> by', torch.softmax(Asq, dim=1), support_clip_probs)
                    _min = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="min")
                    _max = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="max")
                    tmp = einops.rearrange(
                        (tmp - _min) / (_max - _min + 1e-8), 'bsz (hq wq) -> bsz hq wq', hq=_size, wq=_size)
                    _all.append(tmp.unsqueeze(1))
                if combination == "contextfake":
                    _all.append(contextfake)
                if combination == "Aqq@contextfake":
                    _context = contextfake.squeeze(1).view(contextfake.shape[0], -1)
                    tmp = torch.einsum(
                        'bxy,by -> bx', torch.softmax(Aqq, dim=-1), _context)
                    _min = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="min")
                    _max = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="max")
                    tmp = einops.rearrange(
                        (tmp - _min) / (_max - _min + 1e-8), 'bsz (hq wq) -> bsz hq wq', hq=_size, wq=_size)
                    _all.append(tmp.unsqueeze(1))
                if combination == "Asqfake@clip":  
                    _clip = clip_prob.squeeze(1).view(clip_prob.shape[0], -1)
                    tmp = torch.einsum(
                        'bxy,bx -> by', torch.softmax(Asqfake, dim=1), _clip)
                    _min = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="min")
                    _max = einops.reduce(
                        tmp, ' bsz hw -> bsz ()', reduction="max")
                    tmp = einops.rearrange(
                        (tmp - _min) / (_max - _min + 1e-8), 'bsz (hq wq) -> bsz hq wq', hq=_size, wq=_size)
                    _all.append(tmp.unsqueeze(1))
        _all = einops.rearrange(_all, 'n bsz c h w -> bsz (n c) h w')
        return _all

    def cls_seg(self, feat):
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.text_embeddings[:, :, None, None])
        return output

    def build_conv_block(
        self, in_channel, out_channels, kernel_sizes, spt_strides, group=4
    ):
        r"""bulid conv blocks"""
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(
            zip(out_channels, kernel_sizes, spt_strides)
        ):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(
                nn.Conv2d(
                    in_channels=inch,
                    out_channels=outch,
                    kernel_size=ksz,
                    stride=stride,
                    padding=pad,
                    groups=1 if idx == 0 else 16
                )
            )
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)

from mmseg.models import build_segmentor, build_backbone, build_head, build_loss
from model.seghead import seghead
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.utils import Registry
from operator import add
from functools import reduce
import torch
import sys
import numpy as np
import einops
import torch.nn.functional as F
import os
import cv2
import random
import string
sys.path.append('repository/MaskCLIP/mmseg')
from mmseg.ops import resize
import mmcv
HEADS = Registry('head')
HEADS.register_module(seghead)


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = (
        F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w
        + 0.0005
    )
    supp_feat = (
        F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:])
        * feat_h
        * feat_w
        / area
    )
    return supp_feat


backbone_name = 'RN50'
all_stage = True
use_dropout = True
dataset_name = 'voc'

num_class = {

    'voc': 20,
    'coco': 80,
    'fss':1000
}
text_ckpt_name_mapping = {
    "voc":"voc",
    "coco":"stuff",
    "fss":"fss1000"}

CKPT_dict = {
    'RN50': {
        "text_embeddings_path": "repository/MaskCLIP/pretrain/{}_RN50_clip_text.pth".format(text_ckpt_name_mapping[dataset_name]),
        "visual_projs_path": "repository/MaskCLIP/pretrain/RN50_clip_weights.pth",
        "backbone_path": "repository/MaskCLIP/pretrain/RN50_clip.pth",
        "in_channels": 2048,
        "text_channels": 1024,
        "depth": 50,
        "nlayers":[3, 4, 6, 3] if all_stage else [1,1,1,1],
        "use_dropout": use_dropout,
    },
    "RN50x4": {
        "text_embeddings_path": "repository/MaskCLIP/pretrain/{}_RN50x4_clip_text.pth".format(text_ckpt_name_mapping[dataset_name]),
        "visual_projs_path": "repository/MaskCLIP/pretrain/RN50x4_clip_weights.pth",
        "backbone_path": "repository/MaskCLIP/pretrain/RN50x4_clip.pth",
        "in_channels": 2560,
        "text_channels": 640,
        "depth": '50x4',
        "nlayers": [4, 6, 10, 6] if all_stage else [1,1,1,1],
        "use_dropout": use_dropout,
    },
    "RN50x16": {
        "text_embeddings_path": "repository/MaskCLIP/pretrain/{}_RN50x16_clip_text.pth".format(text_ckpt_name_mapping[dataset_name]),
        "visual_projs_path": "repository/MaskCLIP/pretrain/RN50x16_clip_weights.pth",
        "backbone_path": "repository/MaskCLIP/pretrain/RN50x16_clip.pth",
        "in_channels": 3072,
        "text_channels": 768,
        "depth": '50x16',
        "nlayers": [6, 8, 18, 8] if all_stage else [1,1,1,1],
        "use_dropout": use_dropout,
    },
    "RN101": {
        "text_embeddings_path": "repository/MaskCLIP/pretrain/{}_RN101_clip_text.pth".format(text_ckpt_name_mapping[dataset_name]),
        "visual_projs_path": "repository/MaskCLIP/pretrain/RN101_clip_weights.pth",
        "backbone_path": "repository/MaskCLIP/pretrain/RN101_clip.pth",
        "in_channels": 2048,
        "text_channels": 512,
        "depth": 101,
        "nlayers":[3, 4, 23, 3] if all_stage else [1,1,1,1],
        "use_dropout": use_dropout,
    },
    'ViT16': {
        "text_embeddings_path": "repository/MaskCLIP/pretrain/{}_ViT16_clip_text.pth".format(text_ckpt_name_mapping[dataset_name]),
        "visual_projs_path": "repository/MaskCLIP/pretrain/ViT16_clip_weights.pth",
        "backbone_path": "repository/MaskCLIP/pretrain/ViT16_clip.pth",
        "in_channels": 768,
        "text_channels": 512,
        "patch_size": 16,
        "nlayers": [4,4,4,4],
        "use_dropout": use_dropout,
    },
    'ViT32': {
        "text_embeddings_path": "repository/MaskCLIP/pretrain/{}_ViT32_clip_text.pth".format(text_ckpt_name_mapping[dataset_name]),
        "visual_projs_path": "repository/MaskCLIP/pretrain/ViT32_clip_weights.pth",
        "backbone_path": "repository/MaskCLIP/pretrain/ViT32_clip.pth",
        "in_channels": 768,
        "text_channels": 512,
        "patch_size": 32,
        "nlayers": [4, 4, 4, 4],
        "use_dropout": use_dropout,
    },
}



if backbone_name in ['RN50', 'RN50x4', 'RN50x16', 'RN101']:
    _cfg = dict(
        model=dict(
            type='EncoderDecoder',
            backbone=dict(
                type='ResNetClip',
                depth=CKPT_dict[backbone_name]['depth'],
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                contract_dilation=True,
                out_indices=(0, 1, 2, 3),
                stem_channels=96 if backbone_name in ['RN50x4', 'RN50x16'] else 64,
                base_channels=96 if backbone_name in ['RN50x4', 'RN50x16'] else 64,
            ),
            decode_head=dict(
                type='seghead',
                text_categories=num_class[dataset_name],
                text_channels=CKPT_dict[backbone_name]['text_channels'],
                in_channels=CKPT_dict[backbone_name]['in_channels'],
                channels=0,
                num_classes=num_class[dataset_name],
                dropout_ratio=0,
                in_index=3,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                ),
                init_cfg=dict(),
                vit=False,
                text_embeddings_path=CKPT_dict[backbone_name]['text_embeddings_path'],
                visual_projs_path=CKPT_dict[backbone_name]['visual_projs_path'],
                decode_mode='clip_context_affinity_v17', 
                other_args=CKPT_dict[backbone_name]),
        ),
        train_cfg=dict(),
        test_cfg=dict(mode='whole'),
    )
elif backbone_name in ['ViT16', 'ViT32']:
    _cfg = dict(
        model=dict(
            type='EncoderDecoder',
            backbone=dict(
                type='VisionTransformer',
                img_size=(224, 224),
                patch_size=CKPT_dict[backbone_name]['patch_size'],
                patch_bias=False,
                in_channels=3,
                embed_dims=CKPT_dict[backbone_name]['in_channels'],
                num_layers=12,
                num_heads=12,
                mlp_ratio=4,
                out_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                with_cls_token=True,
                output_cls_token=True,
                norm_cfg=dict(type='LN', eps=1e-06),
                act_cfg=dict(type='GELU'),
                patch_norm=False,
                pre_norm=True,
                final_norm=True,
                return_qkv=True,
                interpolate_mode='bicubic',
                num_fcs=2,
                norm_eval=False,
            ),
            decode_head=dict(
                type='seghead',
                text_categories=num_class[dataset_name],
                text_channels=CKPT_dict[backbone_name]['text_channels'],
                in_channels=CKPT_dict[backbone_name]['in_channels'],
                channels=0,
                num_classes=num_class[dataset_name],
                dropout_ratio=0,
                in_index=3,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                ),
                init_cfg=dict(),
                vit=True,
                text_embeddings_path=CKPT_dict[backbone_name]['text_embeddings_path'],
                visual_projs_path=CKPT_dict[backbone_name]['visual_projs_path'],
                decode_mode='clip_context_affinity_v17', 
                other_args=CKPT_dict[backbone_name]
            ),
        ),
        train_cfg=dict(),
        test_cfg=dict(mode='whole'),
    )


class Backbone(torch.nn.Module):
    def __init__(
        self,
    ):
        super(Backbone, self).__init__()

        cfg = mmcv.Config(_cfg)
        assert os.path.exists(CKPT_dict[backbone_name]['backbone_path'])
        tmp = {'checkpoint': CKPT_dict[backbone_name]['backbone_path']}
        cfg.merge_from_dict(tmp)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        self.cfg = cfg
        self.backbone = build_backbone(cfg.model.backbone)
        self.decoder = build_head(cfg.model.decode_head)
        checkpoint = load_checkpoint(
            self.backbone, tmp['checkpoint'], map_location='cpu'
        )
        self.freeze_and_set_to_train_mode()
        self.all_stage = all_stage


    def freeze_and_set_to_train_mode(self):
        """Set the module to training mode."""
        if self.cfg.model.backbone.get('need_dpt', False):
            self.backbone.set_prompt_train_mode(True)
        else:
            self.backbone.eval()
            self.backbone.requires_grad_(False)

        if self.cfg.model.decode_head.vit:
            for param in self.decoder.proj.parameters():
                param.requires_grad = False
        else:  #
            for param in self.decoder.q_proj.parameters():
                param.requires_grad = False
            for param in self.decoder.k_proj.parameters():
                param.requires_grad = False
            for param in self.decoder.v_proj.parameters():
                param.requires_grad = False
            for param in self.decoder.c_proj.parameters():
                param.requires_grad = False


    def forward(self, batch):
        if self.cfg.model.decode_head.decode_mode == 'text_as_classifier':
            _all = self.forward_original_clip_text_as_classifier(batch)
            return _all
        else:
            with torch.no_grad():
                if self.cfg.model.decode_head.vit:
                    batch['query_feat'] = self.backbone(batch['query_img'])
                else :# 
                    if not self.all_stage:
                        batch['query_feat'] = self.backbone(batch['query_img'])
                    else: 
                        batch['query_feat'] = self.get_all_stage_feats(batch['query_img'])

        if self.cfg.model.decode_head.decode_mode.startswith('clip_context_affinity'):
            with torch.no_grad(): 
                if self.cfg.model.decode_head.vit:
                    batch['support_feat'] = self.forward_backbone_support(batch)
                else:
                    if not self.all_stage:
                        batch['support_feat'] = self.forward_backbone_support(batch)
                    else:
                        batch['support_feat'] = self.get_all_stage_feats(
                            einops.rearrange(batch['support_imgs'], 'b n c h w -> (b n) c h w'))
                
                
                batch['clip_probs'] = self.get_clip_probs(batch)
                batch['support_clip_probs'] = self.get_clip_probs( batch, _forwad_type='support')
                batch['context_probs']= self.get_context_probs(batch)
                batch['contextfake'] = self.get_context_probs_fake(batch)
                batch['affinity'] = self.get_affinity(batch)
                batch['Asqfake'] = self.get_Asq_fake(batch)
                torch.cuda.empty_cache()
            out = self.decoder.forward(
                batch
            )  

            return out

    def get_all_stage_feats(self, imgs):
        if backbone_name == 'RN50':
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone_name == 'RN101':
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone_name == 'RN50x16':
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [6, 8, 18, 8]
            self.feat_ids = list(range(0, 41))
        else :
            raise NotImplementedError
        self.lids = reduce(
            add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)

        feats = []
        if backbone_name in ['RN50', 'RN101']:
            bottleneck_ids = reduce(
                add, list(map(lambda x: list(range(x)), self.nlayers))
            )
            if self.backbone.deep_stem:
                feat = self.backbone.stem(imgs)
            else:
                feat = self.backbone.conv1(imgs)
                feat = self.backbone.norm1(feat)
                feat = self.backbone.relu(feat)
            feat = self.backbone.stem_pool(feat)
            
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].forward(feat)
                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())
            
            return feats

    def get_clip_probs(self, batch, _forwad_type='query'):
        _all = self.decoder.forward_original_clip_text_as_classifier(
            batch, _forwad_type)
        clip_probs = _all['logits_by_softmax']
        return clip_probs

    def get_context_probs(self, batch):
        query_feats, support_feats = (batch['query_feat'], batch['support_feat'])

        if self.cfg.model.decode_head.vit:
            query_feats = [item[-1] for item in query_feats]
            support_feats = [item[-1] for item in support_feats]

        cos_similarities = []
        for idx,(query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            a = CKPT_dict[backbone_name]["nlayers"]
            tmp = a[0]+a[1]
            if idx < tmp:
                continue
            else:
                support_feat = torch.relu(input=support_feat)
                support_mask = F.interpolate(
                    batch['support_masks'],
                    size=support_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=True,
                )

                support_feat = (
                    Weighted_GAP(support_feat, support_mask).squeeze(-1).squeeze(-1)
                )

                support_feat = support_feat / support_feat.norm(dim=1, keepdim=True)
                
                query_feat = torch.relu(input=query_feat)
                query_feat = query_feat / query_feat.norm(dim=1, keepdim=True)


                corr = torch.einsum('bchw,bc->bhw', [query_feat, support_feat])
                _max = einops.reduce(corr, 'b h w -> b () ()', reduction='max')
                _min = einops.reduce(corr, 'b h w -> b () ()', reduction='min')
                corr = (corr - _min) / (_max - _min + 1e-8)
                corr = torch.where(torch.isnan(corr.float()), torch.zeros_like(corr), corr)
                cos_similarities.append(corr)
        return cos_similarities

    def get_context_probs_fake(self,batch):
        query_feats, support_feats = (batch['query_feat'], batch['support_feat'])

        if self.cfg.model.decode_head.vit:
            query_feats = [item[-1] for item in query_feats]
            support_feats = [item[-1] for item in support_feats]

        cos_similarities = []
        for idx,(query_feat, support_feat) in  enumerate(zip(query_feats, support_feats)):
            a = CKPT_dict[backbone_name]["nlayers"]
            tmp = a[0]+a[1]
            if idx < tmp:
                continue
            else:
                support_feat = torch.relu(input=support_feat)
                support_mask = F.interpolate(
                    batch['support_clip_probs'],
                    size=support_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=True,
                )

                support_feat = (
                    Weighted_GAP(support_feat, support_mask).squeeze(-1).squeeze(-1)
                )

                support_feat = support_feat / support_feat.norm(dim=1, keepdim=True)
                
                query_feat = torch.relu(input=query_feat)
                query_feat = query_feat / query_feat.norm(dim=1, keepdim=True)


                corr = torch.einsum('bchw,bc->bhw', [query_feat, support_feat])
                _max = einops.reduce(corr, 'b h w -> b () ()', reduction='max')
                _min = einops.reduce(corr, 'b h w -> b () ()', reduction='min')
                corr = (corr - _min) / (_max - _min + 1e-8)
                corr = torch.where(torch.isnan(corr.float()), torch.zeros_like(corr), corr)
            
                cos_similarities.append(corr)
        return cos_similarities



    def get_affinity(self, batch):
        query_feats, support_feats, support_mask = (
            batch['query_feat'],
            batch['support_feat'],
            batch['support_masks'],
        )
        if self.cfg.model.decode_head.vit:
            query_feats = [item[-1] for item in query_feats]
            support_feats = [item[-1] for item in support_feats]

        _affinity = []
        for idx,(query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            a = CKPT_dict[backbone_name]["nlayers"]
            tmp = a[0]+a[1]
            if idx < tmp:
                continue
            else:
                mask = F.interpolate(
                    support_mask,
                    size=support_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=True,
                )
                support_feat = einops.rearrange(
                    torch.relu(support_feat * mask), 'b c h w -> b (h w) c'
                )
                query_feat = einops.rearrange(
                    torch.relu(query_feat), 'b c h w -> b (h w) c'
                )

                A_ss = self.correlation(support_feat, support_feat)
                A_sq = self.correlation(support_feat, query_feat)
                A_qq = self.correlation(query_feat, query_feat)

                _affinity.append([A_ss, A_sq, A_qq])
        return _affinity

    def get_Asq_fake(self, batch):  
        query_feats, support_feats, support_mask = (
            batch['query_feat'],
            batch['query_feat'],
            batch['clip_probs'],
        )
        if self.cfg.model.decode_head.vit:
            query_feats = [item[-1] for item in query_feats]
            support_feats = [item[-1] for item in support_feats]

        _affinity = []
        for idx,(query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            a = CKPT_dict[backbone_name]["nlayers"]
            tmp = a[0]+a[1]
            if idx < tmp:
                continue
            else:
                mask = F.interpolate(
                    support_mask,
                    size=support_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=True,
                )
                support_feat = einops.rearrange(
                    torch.relu(support_feat * mask), 'b c h w -> b (h w) c'
                )
                query_feat = einops.rearrange(
                    torch.relu(query_feat), 'b c h w -> b (h w) c'
                )
                A_sq = self.correlation(support_feat,support_feat)

                _affinity.append(A_sq)
        return _affinity





    def forward_original_clip_text_as_classifier(
        self, batch
    ):  
        with torch.no_grad():
            batch['query_feat'] = self.backbone(batch['query_img'])
            _all = self.decoder.forward_original_clip_text_as_classifier(batch)
        return _all

    def correlation(self, feat1, feat2):
        feat1 = feat1 / (feat1.norm(dim=-1, keepdim=True) + 1e-8)
        feat2 = feat2 / (feat2.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.bmm(feat1, feat2.permute(0, 2, 1))

    def forward_backbone_support(self, batch):
        b = batch['query_img'].shape[0]

        masked_support_imgs = batch['support_imgs']
        support_feat = self.backbone(
            einops.rearrange(masked_support_imgs, 'b n c h w -> (b n) c h w')
        )
        return support_feat


if __name__ == '__main__':
    model = vitmm7()
    print(model)

    img = torch.rand(2, 3, 224, 224)
    batch = dict(
        query_img=img,
        support_imgs=img,
        support_masks=torch.ones(2, 5, 224, 224),
        support_labels=torch.ones(2, 5),
    )
    out = model(batch)
    print(out)

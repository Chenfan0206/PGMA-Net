r""" Hypercorrelation Squeeze training (validation) code """
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
import os
# from model.vit_prompt.vit_mm3_SegTextAsConditionHead import (
#     vitmm3_SegTextAsConditionHead as mymodel,
# )
# from model.vit_prompt.vit_mm4 import (vitmm4 as mymodel,)
# from model.vit_prompt.vit_mm5_dpt import vitmm5 as mymodel
# from model.vit_prompt.vit_mm6_dpt import vitmm6 as mymodel
# from model.vit_prompt.vit_mm7 import vitmm7 as mymodel
# from model.vit_prompt.vit_mm8 import vitmm8 as mymodel\
# from model.vit_prompt.vit_mm9 import vitmm9 as mymodel
from model.vit_prompt.vit_mm10 import vitmm10 as mymodel

from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.vis import Visualizer
from common import utils
from data.dataset import FSSDataset
import numpy as np
from mmcv.runner import load_checkpoint,load_state_dict,save_checkpoint
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_num_threads(2)

voc_classes = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']


coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling', 'tile ceiling', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk', 'dirt', 'door', 'fence', 'marble floor', 'floor', 'stone floor', 'tile floor', 'wood floor', 'flower', 'fog', 'food', 'fruit', 'furniture', 'grass', 'gravel', 'ground', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky', 'skyscraper', 'snow', 'solid', 'stairs', 'stone', 'straw', 'structural', 'table', 'tent', 'textile', 'towel', 'tree', 'vegetable', 'brick wall', 'concrete wall', 'wall', 'panel wall', 'stone wall', 'tile wall', 'wood wall', 'water', 'waterdrops', 'blind window', 'window', 'wood'][:80]





def test_vitclip(model, dataloader, nshot):

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    iou_per_class = {}
    fold = args.fold
    class_per_fold = 20 if args.benchmark=="coco" else 5


    for class_id in dataloader.dataset.class_ids:
        iou_per_class[class_id] = []

    # if args.benchmark=="pascal":
    #     for i in range(class_per_fold*fold, class_per_fold*(fold+1)):
    #         iou_per_class[i] = []
    # else:
    #     for i in range(args.fold, 80,4):
    #         iou_per_class[i] = []

    print(len(iou_per_class))
    print(iou_per_class)

    for idx, batch in enumerate(tqdm(dataloader)):

        batch = utils.to_cuda(batch)
        query_path = batch['query_path'][0]

        with torch.no_grad():
            # pred_logits = model(batch['query_img'])
            # pred_mask = model(img=img, img_metas=img_metas, return_loss=False)
            # pred_logits = model.forward_dummy(img=batch['query_img'])
            
            # pred_logits =model(batch)['pred_logits']
            # pred_logits =model.module.forward_txt_as_classifier(batch)['pred_logits']
            if args.nshot==1:
                pred_mask_01 = model(batch)['pred_mask_01']
            else:
                pred_mask_logits_list = []

                _all_support = {}
                _all_support['support_imgs'] = batch['support_imgs'] # 16 5 3 384 384
                _all_support['support_masks'] = batch['support_masks'] # 16 5 384 384
                _all_support['support_names'] = batch['support_names'] # [(16个name) ()]
                _all_support['support_paths'] = batch['support_paths'] # [(16个paths) ()]
                
                
                if args.benchmark=="pascal":
                    _all_support['support_ignore_idxs'] = batch['support_ignore_idxs'] # # 16 5 384 384

                # 需要逐个输入

                for i in range(args.nshot):
                    batch['support_imgs'] = _all_support['support_imgs'][:,i:i+1,:,:,:]
                    batch['support_masks'] = _all_support['support_masks'][:,i:i+1,:,:]
                    batch['support_names'] = [_all_support['support_names'][i]]
                    batch['support_paths'] = [_all_support['support_paths'][i]]
                    if args.benchmark=="pascal":
                        batch['support_ignore_idxs'] = _all_support['support_ignore_idxs'][:,i:i+1,:,:]
                    pred_logit = model(batch)['pred_logits']
                    pred_logit = torch.softmax(pred_logit, dim=1)
                    pred_mask_logits_list.append(pred_logit)
                pred_mask_logits = sum(pred_mask_logits_list) / len(pred_mask_logits_list)
                pred_mask_01 = torch.argmax(pred_mask_logits, dim=1)


        # pred_mask_c = pred_logits.argmax(dim=1)
        # print(pred_mask_c.max())
        # pred_mask_01 = torch.zeros_like(pred_mask_c)
        # pred_mask_01[pred_mask_c == batch['class_id']] = 1

        pred_mask = pred_mask_01


        assert pred_mask.size() == batch['query_mask'].size()

        # convert pred_mask to int32. 我也不知道为什么要加这个？？？。 但是如果不加，会报错
        # pred_mask = pred_mask.type(torch.int32)
        ## 计算iou
        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)


        # count the class_wise miou, and visuakl the plot

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(
                batch['support_imgs'],
                batch['support_masks'],
                batch['query_img'],
                batch['query_mask'],
                pred_mask,
                batch['class_id'],
                idx,
                area_inter[1].float() / area_union[1].float(),
            )
        
        for class_id, miou in zip (batch['class_id'], area_inter[1].float() / area_union[1].float()):
            iou_per_class[class_id.item()].append(miou.item()*100)


        # if idx == 10:
        #     break
  

    ## print the class_wise miou
    # for i in range(class_per_fold*fold, class_per_fold*(fold+1)):
    for i in dataloader.dataset.class_ids:
        print('class_id: %d, miou: %f' % (i, np.mean(iou_per_class[i])))
        Logger.info(msg='class_id: %d, iou: %f' %
                    (i, np.mean(iou_per_class[i])))

    ## plot the history of miou, each class has a plot in the same figure with different color, and save the figure

    class_names = coco_classes if args.benchmark=="coco" else voc_classes


    iou_lists = []
    labels = []
    # for i in range(class_per_fold*fold, class_per_fold*(fold+1)):
    
    if args.benchmark=="pascal":
        for i in dataloader.dataset.class_ids:
            iou_lists.append(iou_per_class[i])
            labels.append("{}:{} IoU:{:.1f}".format(i, class_names[i],np.mean(iou_per_class[i])))



    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()
    print('*'*80)
    print('fold:{}, nshot:{}, miou:{:.1f}, fb_iou:{:.1f}'.format(args.fold, args.nshot, miou, fb_iou))
    print('*'*80)

    
    bins = np.linspace(0, 100, 20)

    if args.benchmark=="coco":
        plt.figure(figsize=(10, 8))
    if args.benchmark=="pascal":
        plt.figure(figsize=(5, 4))

    plt.figure(figsize=(5, 4))
    plt.hist(iou_lists, bins, label=labels)
    plt.legend(loc='upper left')
    plt.title('Fold:{} mIoU:{:.1f}'.format(args.fold,miou))
    plt.xlabel('IoU (%)')
    plt.ylabel('Count Number')
    # plt.show()

    # save the figure
    # save_path = Visualizer.vis_path.replace('test_vis/', 'histogram_fold_{}_shot_{}.png'.format(args.fold, args.nshot))
    save_path = Visualizer.vis_path.replace('test_vis/', 'histogram_fold_{}_shot_{}.pdf'.format(args.fold, args.nshot))
    print(save_path)
    print(Visualizer.vis_path)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # if args.benchmark=="coco":
    #     plt.legend(fontsize='small', ncol=2)
    plt.savefig(save_path)
    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(
        description='Pytorch Implementation of seg via clip'
    )
    parser.add_argument('--datapath', type=str, default='./Datasets_HSN')
    parser.add_argument(
        '--benchmark',
        type=str,
        default='pascal',
        choices=['pascal', 'coco', 'fss', 'pascal_extend', 'coco_extend', 'fss_extend'],
    )
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--strong_aug', default=False, type=bool)

    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization

    model = mymodel()

    # model.eval()
    for param in model.parameters():
        param.requires_grad = False

    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    
    if args.load != '':
        Logger.info('Loading model from %s...' % args.load)
        print('Loading model from %s...' % args.load)
        model.load_state_dict(torch.load(args.load), strict=False)

    else:
        Logger.info('No checkpoint found. Initializing model from scratch.')
        
    model.to(device)

    # load form checkpoint
    

    # Helper classes (for training) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, vis_path=os.path.join(Logger.logpath, 'test_vis/'))

    # Dataset initialization
    FSSDataset.initialize(
        img_size=384, datapath=args.datapath, use_original_imgsize=False,debug=args.debug, strong_aug=args.strong_aug
    )
    # dataloader_trn = FSSDataset.build_dataloader(
    #     args.benchmark, args.bsz, args.nworker, args.fold, 'trn'
    # )
    dataloader_val = FSSDataset.build_dataloader(
        args.benchmark, args.bsz, args.nworker, args.fold, 'val',args.nshot
    )

     # Test HSNet
    with torch.no_grad():
        test_miou, test_fb_iou = test_vitclip(model, dataloader_val, 1)
        # test_miou, test_fb_iou = test_vitclip(model, dataloader_trn, 1)

    Logger.info(
        'Fold %d mIoU: %5.1f \t FB-IoU: %5.1f'
        % (args.fold, test_miou.item(), test_fb_iou.item())
    )

    
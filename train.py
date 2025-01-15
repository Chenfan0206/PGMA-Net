r"""Visual and Textual Prior Guided Mask Assemble for Few-Shot Segmentation and Beyond """
import argparse
import torch.optim as optim
import torch.nn as nn
import torch
torch.autograd.set_detect_anomaly(True)
from model.backbone import Backbone as mymodel
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from mmcv.runner import load_checkpoint, load_state_dict, save_checkpoint
import time
import datetime

torch.set_num_threads(4)
def train(epoch, model, dataloader, optimizer, training):
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.freeze_and_set_to_train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        _all = model(batch)
        loss = _all['loss']
        pred_mask = _all['pred_mask_01']
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
            average_meter.update(
                area_inter, area_union, batch['class_id'], loss.detach().clone()
            )
            average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visual and Textual Prior Guided Mask Assemble for Few-Shot Segmentation and Beyond'
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
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--strong_aug', default=False, type=bool)
    args = parser.parse_args()
    Logger.initialize(args, training=True)
    model = mymodel()
    if args.load != '':
        Logger.info('Loading model from %s' % args.load)
        load_checkpoint(model, args.load, map_location='cpu')
    trainable_num = 0
    print('the following parameters will be trained:')
    for name, p in model.named_parameters():
        if p.requires_grad:
            n_param = model.state_dict()[name].view(-1).size(0)
            print(name, n_param)
            trainable_num += n_param
    decoder_affinity_trainable_num = 0
    for name, p in model.named_parameters():
        if name.startswith('decoder.affinity_blocks'):
            if p.requires_grad:
                n_param = model.state_dict()[name].view(-1).size(0)
                print(name, n_param)
                decoder_affinity_trainable_num += n_param
    print('decoder_affinity_trainable_num:', decoder_affinity_trainable_num)

    decoder_other_trainable_num = 0
    for name, p in model.named_parameters():
        if not name.startswith('decoder.affinity_blocks'):
            if p.requires_grad:
                n_param = model.state_dict()[name].view(-1).size(0)
                print(name, n_param)
                decoder_other_trainable_num += n_param
    print('decoder_other_trainable_num:', decoder_other_trainable_num)

    print('=============================================' * 10)
    print('Total trainable parameters:', trainable_num)

    print('the following parameters will not be trained:')
    print('=============================================' * 10)
    fixed_num = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            n_param = model.state_dict()[name].view(-1).size(0)
            print(name, n_param)
            fixed_num += n_param
    print('Total fixed parameters:', fixed_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    Evaluator.initialize()
    FSSDataset.initialize(
        img_size=384, datapath=args.datapath, use_original_imgsize=False, debug=args.debug, strong_aug=args.strong_aug
    )
    dataloader_trn = FSSDataset.build_dataloader(
        args.benchmark, args.bsz, args.nworker, args.fold, 'trn'
    )
    dataloader_val = FSSDataset.build_dataloader(
        args.benchmark, args.bsz, args.nworker, args.fold, 'val'
    )
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.niter):
        start_time = time.time()
        trn_loss, trn_miou, trn_fb_iou = train(
            epoch, model, dataloader_trn, optimizer, training=True
        )
        current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        Logger.info("current timestamp: %s, epoch: %d, epoch train time: %f" % (current_timestamp, epoch, time.time() - start_time))
        start_time = time.time()
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(
                epoch, model, dataloader_val, optimizer, training=False
            )
        current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        Logger.info("current timestamp: %s, epoch: %d, epoch val time: %f" % (current_timestamp, epoch, time.time() - start_time))
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)
            print('best_val_miou: ', best_val_miou, 'at epoch: ', epoch, 'fold: ', args.fold)
        Logger.tbd_writer.add_scalars(
            'data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch
        )
        Logger.tbd_writer.add_scalars(
            'data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch # type: ignore
        )
        Logger.tbd_writer.add_scalars(
            'data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch # type: ignore
        )
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')

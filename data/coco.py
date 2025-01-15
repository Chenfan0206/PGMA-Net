r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import cv2
import albumentations as A



class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize,debug):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def mask2box(self, mask):
        # mask: numpy array, 1 is object, 0 is background
        # Returns: new mask with the same size as the input mask, but the bounding box of the object is 1, the rest is 0
        if mask.sum() == 0:
            return mask
        else:
            mask = mask.astype(np.uint8)

            ## perform the all max2box operation
            tmp = mask
            h_axis = np.sum(tmp, axis=1)
            h_ind = np.nonzero(h_axis)
            w_axis = np.sum(tmp, axis=0)
            w_ind = np.nonzero(w_axis)
            box = [h_ind[0][0], h_ind[0][-1], w_ind[0][0],
                   w_ind[0][-1]]  # height0.height1,width0,width1
            new_mask = np.zeros_like(mask)
            new_mask[box[0]:box[1], box[2]:box[3]] = 1

            return new_mask

    def disturb_mask(self, mask):
        if mask.sum() == 0:
            return mask
        else:
            mask = mask.astype(np.uint8)
            kernel_size = np.random.choice([7, 9, 11, 13])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilate_times = np.random.choice([4, 5, 6, 7])
            dilate_or_erode = np.random.choice([0, 1])

            if dilate_or_erode == 0:
                tmp = cv2.dilate(mask, kernel, iterations=dilate_times)
            else:
                tmp = cv2.erode(mask, kernel, iterations=dilate_times)
            new_mask = tmp
            return new_mask

    def disturb_image(self, image):

        distrub_transform_2 = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.GaussNoise(var_limit=(40.0, 60.0), mean=0, p=0.5),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2),
                                  elementwise=True, per_channel=True, p=0.5),
        ])

        distrub_transform_3 = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.GaussNoise(var_limit=(40.0, 60.0), mean=0, p=0.5),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2),
                                  elementwise=True, per_channel=True, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                       b_shift_limit=15, p=0.2),
            A.ChannelShuffle(p=0.2),
            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
            A.InvertImg(p=0.2),
            A.ToGray(p=0.2),
        ])

        disturb_transform = distrub_transform_2
        new_image = disturb_transform(image=image)['image']
        return new_image


    def __getitem__(self, idx):

        what_image_to_disturb = []
        # what_image_to_disturb = ['support']
        # what_image_to_disturb = ['query']
        # what_image_to_disturb = ['support','query']
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()
        if 'query' in what_image_to_disturb:
            query_img = self.disturb_image(query_img)
        if 'support' in what_image_to_disturb:
            support_imgs = [self.disturb_image(support_img) for support_img in support_imgs]
        # query_img = self.transform(query_img)
        _tmp = self.transform(image=query_img, mask=query_mask)
        query_img = _tmp['image']
        query_mask = _tmp['mask']
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        # support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        _tmp = [self.transform(image=support_img, mask=support_mask) for support_img,
                support_mask in zip(support_imgs, support_masks)]
        support_imgs = torch.stack([item['image'] for item in _tmp])
        support_masks = [item['mask'] for item in _tmp]
        
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,
                 'query_path': os.path.join(self.base_path, query_name),

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_paths': [os.path.join(self.base_path, name) for name in support_names],
                 'class_id': torch.tensor(class_sample)}


        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        with open('./data/splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png'))
        return mask
    

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        # query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_img = cv2.cvtColor(cv2.imread(os.path.join(self.base_path, query_name)), cv2.COLOR_BGR2RGB)
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            # support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_imgs.append(cv2.cvtColor(cv2.imread(os.path.join(self.base_path, support_name)), cv2.COLOR_BGR2RGB))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            
            # support_mask = self.mask2box(support_mask)
            # support_mask = self.disturb_mask(support_mask)
            # support_mask = np.ones_like(support_mask)
            # support_mask = np.zeros_like(support_mask)
            
            
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize


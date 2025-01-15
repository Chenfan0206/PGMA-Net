r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import copy
import albumentations as A
import cv2


class DatasetPASCAL(Dataset):
    def __init__(
        self, datapath, fold, transform, split, shot, use_original_imgsize, debug
    ):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        # self.pre_load = True
        self.if_predload = not debug

        if self.if_predload:
            self.preload()

        self.load_base_clabel = False

    def preload_one(self, img_name):
        return {
            img_name: {
                "img": copy.deepcopy(self.read_img(img_name)),
                'mask': copy.deepcopy(self.read_mask(img_name)),
            }
        }

    def preload(self):
        print('Preloading images and masks...')
        self.preloadings = {}
        import time

        start = time.time()
        name_list = [item[0] for item in self.img_metadata]
        using_mp = False
        if using_mp:
            with Pool(32) as p:
                tmp = p.map(self.preload_one, name_list)
            self.preloadings.update(tmp)
        else:
            for img_name in tqdm(name_list):
                self.preloadings.update(self.preload_one(img_name))
        print('Preloading done in {:.2f}s'.format(time.time() - start))
        print('Preloaded {} images'.format(len(self.preloadings)))

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def convert_to_mask_c_0_to_21_ignore_novel(self, mask_c):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        need_ignore = class_ids_val if self.split == 'trn' else class_ids_trn

        for (
            novel_id
        ) in (
            need_ignore
        ): 
            mask_c[
                mask_c == novel_id + 1
            ] = 0  
        return mask_c

    def mask2box(self,mask):
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
            box = [h_ind[0][0], h_ind[0][-1], w_ind[0][0], w_ind[0][-1]]  # height0.height1,width0,width1
            new_mask = np.zeros_like(mask)
            new_mask[box[0]:box[1],box[2]:box[3]] = 1
            
            return new_mask
        

    def disturb_mask(self,mask):
        if mask.sum() == 0:
            return mask
        else:
            mask = mask.astype(np.uint8)

            # kernel_size = np.random.choice([3,5,7])
            # kernel_size = np.random.choice([3,5,7,9])
            kernel_size = np.random.choice([7,9,11,13])
            kernel = np.ones((kernel_size,kernel_size),np.uint8)

            # dilate_times = np.random.choice([2,3,4])
            # dilate_times = np.random.choice([2,3,4,5])
            dilate_times = np.random.choice([4,5,6,7])

            dilate_or_erode = np.random.choice([0,1])

            if dilate_or_erode == 0:
                tmp = cv2.dilate(mask,kernel,iterations = dilate_times)
            else:
                tmp = cv2.erode(mask,kernel,iterations = dilate_times)

            new_mask = tmp

            # ## save the mask and new_mask for debug
            # save_dir = 'logs/debug/mask2box/dilate_erode'
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            
            # random_name = str(np.random.randint(100000))

            # cv2.imwrite(os.path.join(save_dir,random_name+'_mask.png'),mask*255)
            # cv2.imwrite(os.path.join(save_dir,random_name+'_new_mask.png'),new_mask*255)
            # print('save mask and new_mask to {}'.format(save_dir))
            

            return new_mask

    def disturb_image(self,image):
        
        distrub_transform_2 = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.GaussNoise(var_limit=(40.0, 60.0), mean=0, p=0.5),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), elementwise=True, per_channel=True, p=0.5),
            ])
        
        distrub_transform_3 = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=(1, 4), p=0.5),
            A.GaussNoise(var_limit=(40.0, 60.0), mean=0, p=0.5),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), elementwise=True, per_channel=True, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15,b_shift_limit=15, p=0.2),
            A.ChannelShuffle(p=0.2),
            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
            A.InvertImg(p=0.2),
            A.ToGray(p=0.2),
            ])
        

        
        disturb_transform = distrub_transform_2
        # disturb_transform = distrub_transform_2_add_noise
        # disturb_transform = distrub_transform_3_de_noise
        # disturb_transform = distrub_transform_4
        new_image = disturb_transform(image=image)['image']

        # ## save the image and new_image for debug
        # save_dir = 'logs/debug/disturb_image_4'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # random_name = str(np.random.randint(100000))


        # # first convert the rgb to bgr
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join(save_dir,random_name+'_image.png'),image)
        # cv2.imwrite(os.path.join(save_dir,random_name+'_new_image.png'),new_image)

        # print('save image and new_image to {}'.format(save_dir))


        return new_image




    def __getitem__(self, idx):
        what_image_to_disturb =[]
        # what_image_to_disturb = ['support']
        # what_image_to_disturb = ['query']
        # what_image_to_disturb = ['support','query']

        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        (
            query_img,
            query_cmask,
            support_imgs,
            support_cmasks,
            org_qry_imsize,
        ) = self.load_frame(query_name, support_names)

        if 'query' in what_image_to_disturb:
            query_img = self.disturb_image(query_img)
        if 'support' in what_image_to_disturb:
            support_imgs = [self.disturb_image(support_img) for support_img in support_imgs]


        # query_img = self.transform(query_img)
        _tmp = self.transform(image=query_img,mask=query_cmask)
        query_img = _tmp['image']
        query_cmask = _tmp['mask']
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(
                query_cmask.unsqueeze(0).unsqueeze(0).float(),
                query_img.size()[-2:],
                mode='nearest',
            ).squeeze()


        if self.load_base_clabel:
            query_mask_c_0_to_21_ignore_novel = self.convert_to_mask_c_0_to_21_ignore_novel(
                query_cmask.clone()
            ) 
        query_mask, query_ignore_idx = self.extract_ignore_idx(
            query_cmask.float(), class_sample
        )


        _tmp = [self.transform(image=support_img, mask=support_cmask) for support_img,
                support_cmask in zip(support_imgs, support_cmasks)]
        support_imgs = torch.stack([item['image'] for item in _tmp])
        support_cmasks = torch.stack([item['mask'] for item in _tmp])
        

        support_masks = []
        support_ignore_idxs = []
        support_masks_c_0_to_21_ignore_novel_lists = []
        for scmask in support_cmasks:
            scmask = F.interpolate(
                scmask.unsqueeze(0).unsqueeze(0).float(),
                support_imgs.size()[-2:],
                mode='nearest',
            ).squeeze()

            
            if self.load_base_clabel:
                support_mask_c_0_to_21_ignore_novel = (
                    self.convert_to_mask_c_0_to_21_ignore_novel(scmask.clone())
                )

                support_masks_c_0_to_21_ignore_novel_lists.append(
                    support_mask_c_0_to_21_ignore_novel
                )

            support_mask, support_ignore_idx = self.extract_ignore_idx(
                scmask, class_sample
            )


            ### here add noise to the support mask, or perform mask2box
            # support_mask = self.mask2box(support_mask.numpy())
            # support_mask = self.disturb_mask(support_mask.numpy())
            # support_mask = torch.from_numpy(support_mask).float()

            # support_mask = torch.ones_like(support_mask)


            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)

        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)
        
        if self.load_base_clabel:
            support_masks_c_0_to_21_ignore_novel = torch.stack(
                support_masks_c_0_to_21_ignore_novel_lists
            )

        # batch = {'query_img': query_img,
        #          'query_mask': query_mask,
        #          'query_name': query_name,
        #          'query_ignore_idx': query_ignore_idx,

        #          'org_query_imsize': org_qry_imsize,

        #          'support_imgs': support_imgs,
        #          'support_masks': support_masks,
        #          'support_names': support_names,
        #          'support_ignore_idxs': support_ignore_idxs,

        #          'class_id': torch.tensor(class_sample)}
        if self.load_base_clabel:
            batch = {
                'query_img': query_img,
                'query_mask': query_mask,
                'query_name': query_name,
                'query_ignore_idx': query_ignore_idx,
                'query_path': os.path.join(self.img_path, query_name) + '.jpg',
                'org_query_imsize': org_qry_imsize,
                'support_imgs': support_imgs,
                'support_masks': support_masks,
                'support_names': support_names,
                'support_ignore_idxs': support_ignore_idxs,
                'class_id': torch.tensor(class_sample),
                # 'ex_paths': ex_paths,
                # 'ex_images': ex_images,
                "query_mask_c_0_to_21_ignore_novel": query_mask_c_0_to_21_ignore_novel,
                "support_masks_c_0_to_21_ignore_novel": support_masks_c_0_to_21_ignore_novel,
            }
        else:
            batch = {
                'query_img': query_img,
                'query_mask': query_mask,
                'query_name': query_name,
                'query_ignore_idx': query_ignore_idx,
                'query_path': os.path.join(self.img_path, query_name) + '.jpg',
                'org_query_imsize': org_qry_imsize,
                'support_imgs': support_imgs,
                'support_masks': support_masks,
                'support_names': support_names,
                'support_paths': [os.path.join(self.img_path, name) + '.jpg' for name in support_names],
                'support_ignore_idxs': support_ignore_idxs,
                'class_id': torch.tensor(class_sample),
                # 'ex_paths': ex_paths,
                # 'ex_images': ex_images,
            }

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):

        if not self.if_predload:
            query_img = self.read_img(query_name)
            query_mask = self.read_mask(query_name)
            support_imgs = [self.read_img(name) for name in support_names]
            support_masks = [self.read_mask(name) for name in support_names]
        else:
            query_img = self.preloadings[query_name]['img']
            query_mask = self.preloadings[query_name]['mask']
            support_imgs = [self.preloadings[name]['img'] for name in support_names]
            support_masks = [self.preloadings[name]['mask'] for name in support_names]
        org_qry_imsize = query_img.size
        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize


    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png'))
        return mask
    

    def read_img(self, img_name):
        r"""Return RGB image in cv2"""
        return cv2.cvtColor(cv2.imread(os.path.join(self.img_path, img_name) + '.jpg'), cv2.COLOR_BGR2RGB)

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(
                self.img_metadata_classwise[class_sample], 1, replace=False
            )[0]
            if query_name != support_name:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):
        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join(
                'data/splits/pascal/%s/fold%d.txt' % (split, fold_id)
            )
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [
                [data.split('__')[0], int(data.split('__')[1]) - 1]
                for data in fold_n_metadata
            ]
            return fold_n_metadata

        img_metadata = []
        if (
            self.split == 'trn'
        ):  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif (
            self.split == 'val'
        ):  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise

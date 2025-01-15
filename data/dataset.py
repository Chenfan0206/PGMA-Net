r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class FSSDataset:
    @classmethod
    def initialize(
        cls, img_size, datapath, use_original_imgsize, ex_num=8, ex_way="first",debug=False,strong_aug=False
    ):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS
        }

        cls.img_mean = [0.48145466, 0.4578275, 0.40821073]
        cls.img_std = [0.26862954, 0.26130258, 0.27577711]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        if strong_aug:
            print("using strong augmentation")
            cls.train_transform = A.Compose([
                A.RandomResizedCrop(height=400, width=400, p=1.0),
                # A.RandomCrop(height=512, width=512, p=1.0),
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=(-30, 30), border_mode=cv2.BORDER_REFLECT_101, p=0.2),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.2),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.2)
                ]),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.2),
                    A.IAAAdditiveGaussianNoise(loc=0, scale=(
                        2.5500000000000003, 12.75), per_channel=False, p=0.2)
                ]),
                A.OneOf([
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.2),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2)
                ]),
                A.OneOf([
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
                    A.CLAHE(clip_limit=4.0, p=0.2),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
                ],p=0.2),
                A.Resize(height=img_size, width=img_size, p=1.0),
                A.Normalize(mean=cls.img_mean, std=cls.img_std),
                ToTensorV2(),
            ])
        else:
            cls.train_transform = A.Compose([
                A.HorizontalFlip(p=0.4),
                A.VerticalFlip(p=0.4),
                A.Rotate(limit=(-30, 30), border_mode=cv2.BORDER_REFLECT_101, p=0.4),
                A.RandomResizedCrop(img_size, img_size,scale=(0.8, 1.0), ratio=(0.8, 1.2), p=0.4),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.2),
                A.Resize(height=img_size, width=img_size, p=1.0),
                A.Normalize(mean=cls.img_mean, std=cls.img_std),
                ToTensorV2(),
            ])

        cls.val_transform = A.Compose([
            A.Resize(height=img_size, width=img_size, p=1.0),
            A.Normalize(mean=cls.img_mean, std=cls.img_std),
            ToTensorV2(),
        ])


        cls.ex_num = ex_num
        cls.ex_way = ex_way
        cls.debug = debug

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        # shuffle = True

        nworker = nworker if split == 'trn' else 0
        transform = cls.train_transform if split == 'trn' else cls.val_transform

        dataset = cls.datasets[benchmark](
            cls.datapath,
            fold=fold,
            transform=transform,
            split=split,
            shot=shot,
            use_original_imgsize=cls.use_original_imgsize,
            # ex_num=cls.ex_num,
            # ex_way=cls.ex_way,
            debug=cls.debug,
        )
        dataloader = DataLoader(
            dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker
        )

        return dataloader

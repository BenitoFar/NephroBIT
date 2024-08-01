import os
import yaml
from glob import glob

import random
import numpy as np

import torch

from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ResizeWithPadOrCropd,
    ScaleIntensityRangeD,
    RandFlipd,
    SelectItemsd,
    OneOf, 
    RandGaussianSmoothd, 
    MedianSmoothd, 
    RandGaussianNoised
)
from monai.networks.nets import UNet, UNETR, SwinUNETR, DynUNet

import cv2


def save_mask(image, output_path):
    cv2.imwrite(output_path, np.moveaxis(image, 0, 1))


def load_config(cfg):
    with open(cfg, 'r') as file:
        config = yaml.safe_load(file)
    return config


def prepare_data(datadir):
    """prepare data list"""

    images = sorted(glob(os.path.join(datadir, "**/*img*"), recursive=True)) 
    masks = sorted(glob(os.path.join(datadir, "**/*mask*"), recursive = True))
    #remove all elements that are not files (directories)
    images = [x for x in images if os.path.isfile(x)]
    masks = [x for x in masks if os.path.isfile(x)]
    #check if the number of images and masks is the same
    assert len(images) == len(masks)
    
    print('Number of images:', len(images))
    
    class_list = ['56Nx', 'DN', 'normal', 'NEP25']
    index = [i for i, string in enumerate(os.path.dirname(images[0]).split('/')) if string in class_list][0]
        
    data_list = [
        {"img": _image, "label": _label, 'case_class': os.path.dirname(_image).split('/')[index], 'case_id': os.path.dirname(_image).split('/')[index+1], 'img_path': _image, 'label_path': _label}
        for _image, _label in zip(images, masks)
    ]
    return data_list


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def get_transforms(cfg, phase):
    if phase == 'train':
        if 'CropPosNeg' in cfg['preprocessing']['image_preprocess'] and 'ResizeLargerPatch' not in cfg['preprocessing']['image_preprocess'] and 'FlipRot' not in cfg['preprocessing']['image_preprocess']:
            train_transforms = Compose(
                    [
                    LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                    EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                    EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                    RandCropByPosNegLabeld(keys=["img", "label"], label_key="label", spatial_size= cfg['preprocessing']['roi_size'], pos=3, neg=1, num_samples=cfg['preprocessing']['num_samples_per_image']),
                    OneOf(
                        transforms=[
                            RandGaussianSmoothd(keys=["img"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
                            MedianSmoothd(keys=["img"], radius=1),
                            RandGaussianNoised(keys=["img"], prob=1.0, std=0.05),
                        ]
                    ),   
                    AsDiscreted(keys="label", threshold= 1, dtype=torch.uint8),
                    ScaleIntensityRangeD(keys=("img"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0, clip=True), 
                    # TorchVisiond(
                    #             keys=["img"],
                    #             name="ColorJitter",
                    #             brightness=(229 / 255.0, 281 / 255.0),
                    #             contrast=(0.95, 1.10),
                    #             saturation=(0.8, 1.2),
                    #             hue=(-0.04, 0.04),
                    #         ),
                    SelectItemsd(keys=("img", "label", "case_id", "case_class", "img_path", "label_path")),
                    ]
                )
        elif 'CropPosNeg' in cfg['preprocessing']['image_preprocess'] and 'ResizeLargerPatch' in cfg['preprocessing']['image_preprocess'] and 'FlipRot' not in cfg['preprocessing']['image_preprocess']:
            train_transforms = Compose(
                    [
                    LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                    EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                    EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                    OneOf( 
                          transforms = [
                            RandCropByPosNegLabeld(
                                    keys=["img", "label"], label_key="label", spatial_size= cfg['preprocessing']['roi_size'], pos=3, neg=1, num_samples=cfg['preprocessing']['num_samples_per_image']),
                            Compose(
                                [
                                # CropForegroundd(keys=("img", "label"), source_key="img", spatial_size=(1024, 1024)), 
                                RandCropByPosNegLabeld(
                                    keys=["img", "label"], label_key="label", spatial_size= (1024,1024), pos=3, neg=1, num_samples=cfg['preprocessing']['num_samples_per_image']),
                                ResizeWithPadOrCropd(keys=("img", "label"), spatial_size=cfg['preprocessing']['roi_size']),]
                            )
                            ],
                          weights=[0.9,0.1]
                    ),
                    
                    OneOf(
                        transforms=[
                            RandGaussianSmoothd(keys=["img"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
                            MedianSmoothd(keys=["img"], radius=1),
                            RandGaussianNoised(keys=["img"], prob=1.0, std=0.05),
                        ]
                    ),   
                    AsDiscreted(keys="label", threshold= 1, dtype=torch.uint8),
                    ScaleIntensityRangeD(keys=("img"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0, clip=True), 
                    SelectItemsd(keys=("img", "label", "case_id", "case_class", "img_path", "label_path")),
                    ]
                )
        elif 'CropPosNeg' in cfg['preprocessing']['image_preprocess'] and 'ResizeLargerPatch' in cfg['preprocessing']['image_preprocess'] and 'FlipRot' in cfg['preprocessing']['image_preprocess']:
            train_transforms = Compose(
                    [
                    LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                    EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                    EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                    OneOf( 
                          transforms = [
                            RandCropByPosNegLabeld(
                                    keys=["img", "label"], label_key="label", spatial_size= cfg['preprocessing']['roi_size'], pos=3, neg=1, num_samples=cfg['preprocessing']['num_samples_per_image']),
                            Compose(
                                [
                                # CropForegroundd(keys=("img", "label"), source_key="img", spatial_size=(1024, 1024)), 
                                RandCropByPosNegLabeld(
                                    keys=["img", "label"], label_key="label", spatial_size= (1024,1024), pos=3, neg=1, num_samples=cfg['preprocessing']['num_samples_per_image']),
                                ResizeWithPadOrCropd(keys=("img", "label"), spatial_size=cfg['preprocessing']['roi_size']),]
                            )
                            ],
                          weights=[0.9,0.1]
                    ),
                    RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=0),
                    RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=1),
                    RandRotate90d(keys=("img", "label"), prob=0.5, spatial_axes=(0, 1)),
                    OneOf(
                        transforms=[
                            RandGaussianSmoothd(keys=["img"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
                            MedianSmoothd(keys=["img"], radius=1),
                            RandGaussianNoised(keys=["img"], prob=1.0, std=0.05),
                        ]
                    ),   
                    AsDiscreted(keys="label", threshold= 1, dtype=torch.uint8),
                    ScaleIntensityRangeD(keys=("img"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0, clip=True), 
                    SelectItemsd(keys=("img", "label", "case_id", "case_class", "img_path", "label_path")),
                    ]
                )
        else:
            raise ValueError(f"Unknown image_preprocess: {cfg['preprocessing']['image_preprocess']}")
        train_transforms.set_random_state(seed=cfg['seed'])
        return train_transforms
        
    elif phase == 'val' or phase == 'test':
            val_transforms = Compose(
                    [
                    LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                    EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                    EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                    AsDiscreted(keys="label", threshold= 1, dtype=torch.uint8),
                    ScaleIntensityRangeD(keys=("img"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0, clip=True),
                    SelectItemsd(keys=("img", "label","case_id", "case_class", "img_path", "label_path")),
                    ]
                )
            val_transforms.set_random_state(seed=cfg['seed'])
            return val_transforms
    elif phase == 'ensemble':
            val_transforms = Compose(
                    [
                        LoadImaged(keys=["image", "label"], dtype=torch.uint8),
                        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
                        EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                        AsDiscreted(keys="label", threshold= 1, dtype=torch.uint8),
                        ScaleIntensityRangeD(keys=("image"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0, clip=True),
                        SelectItemsd(keys=("image", "label","case_id", "case_class", "img_path", "label_path")),
                        ]
                    )
            val_transforms.set_random_state(seed=cfg['seed'])
            return val_transforms
    elif phase == 'generate_patches':
            transforms = Compose(
                    [
                    LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                    EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                    EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                    RandCropByPosNegLabeld(
                        keys=["img", "label"], label_key="label", spatial_size= cfg['preprocessing']['roi_size'], pos=1, neg=0, num_samples=cfg['preprocessing']['num_samples_per_image'], allow_smaller = False
                    ),
                    AsDiscreted(keys="label", threshold= 1, dtype=torch.uint8),
                    SelectItemsd(keys=("img", "label", "case_id", "case_class", "img_path", "label_path")),
                    ]
                )
            transforms.set_random_state(seed=cfg['seed'])
            return transforms
    else:
        raise ValueError(f"Unknown phase: {phase}")

def get_model(cfg, pretrained_path=None):
    device = torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu")
    models = []

    def create_model(model_name, model_params):
        if model_name == "Unet":
            return UNet(
                spatial_dims=model_params['spatial_dims'],
                in_channels=model_params['in_channels'],
                out_channels=model_params['out_channels'],
                channels=model_params['f_maps_channels'],
                strides=model_params['strides'],
                num_res_units=model_params['num_res_units'],
            ).to(device)
        elif model_name == "UNETR":
            return UNETR(
                in_channels=model_params['in_channels'],
                out_channels=model_params['out_channels'],
                img_size=cfg['preprocessing']['roi_size'],
                spatial_dims=model_params['spatial_dims'],
                feature_size=model_params['feature_size'],
                hidden_size=model_params['hidden_size'],
                mlp_dim=model_params['mlp_dim'],
                num_heads=model_params['num_heads'],
            ).to(device)
        elif model_name == "SwinUNETR":
            return SwinUNETR(
                img_size=cfg['preprocessing']['roi_size'],
                in_channels=model_params['in_channels'],
                out_channels=model_params['out_channels'],
                spatial_dims=model_params['spatial_dims'],
                depths=model_params['depths'],
                num_heads=model_params['num_heads'],
                feature_size=model_params['feature_size'],
                use_v2=model_params['use_v2'],
            ).to(device)
        elif model_name == "DynUNet":
            kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
            strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
            return DynUNet(
                spatial_dims=model_params['spatial_dims'],
                in_channels=model_params['in_channels'],
                out_channels=model_params['out_channels'],
                kernel_size=kernels,
                strides=strides,
                upsample_kernel_size=strides[1:],
                norm_name="instance",
                deep_supervision=model_params['deep_supervision'],
                deep_supr_num = model_params['deep_supr_num'],
                res_block=model_params['res_block'],
            ).to(device)
        else:
            raise ValueError(f"Model {model_name} not implemented")

    model_names = cfg['model']['name']
    model_params = cfg['model']['params']
    
    if isinstance(model_names, list) and isinstance(model_params, list):
        if len(model_names) != len(model_params):
            raise ValueError("Length of model names and model params lists must be the same.")
        models = [create_model(name, params) for name, params in zip(model_names, model_params)]
    else:
        models = [create_model(model_names, model_params)]

    if pretrained_path is not None:
        if isinstance(pretrained_path, list):
            for i, model_path in enumerate(pretrained_path):
                models[i].load_state_dict(torch.load(model_path)['model_state_dict'])
                print(f"Model {i+1} loaded from {model_path}")
        else:
            models[0].load_state_dict(torch.load(pretrained_path)['model_state_dict'])
            print(f"Model loaded from {pretrained_path}")

    for model in models:
        model.eval()

    if len(models) == 1:
        return models[0]
    else:
        return models
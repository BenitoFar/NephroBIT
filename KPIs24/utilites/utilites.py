import numpy as np
import yaml
import os
from glob import glob
import random
import torch
from matplotlib import pyplot as plt
from monai.apps.nuclick.transforms import SplitLabelMined
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    Lambdad,
    ComputeHoVerMaps,
    ScaleIntensityRangeD,
    RandFlipd,
    SelectItemsd,
    LabelToMaskd,
    Resized,
    OneOf, 
    RandGaussianSmoothd, 
    MedianSmoothd, 
    RandGaussianNoised
)
import monai
from skimage import measure

def show_image(image, label, predictions = None, filename = None):
    # print(f"Image: {image.shape}; Label: {label.shape}")

    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, 2)
        label = np.moveaxis(label, 0, 2)
        if predictions is not None:
            predictions = np.moveaxis(predictions, 0, 2)
            
    fig = plt.figure(figsize=(12, 6))
    if predictions is not None:
        n_subplots = 3
    else:
        n_subplots = 2    
    
    plt.subplot(1, n_subplots, 1)  
    plt.title("image")
    plt.imshow(image, cmap="gray")

    # if label is not None:
    #     masked = np.ma.masked_where(label == 0, label)
    #     plt.imshow(masked, "jet", interpolation="none", alpha=0.5)
    
    # plt.colorbar()

    if label is not None:
        plt.subplot(1, n_subplots, 2)
        plt.title("label")
        plt.imshow(label, cmap="gray", interpolation="none")
        # plt.colorbar()
    
    if predictions is not None:
        plt.subplot(1, n_subplots, 3)
        plt.title("prediction")
        plt.imshow(predictions, cmap="gray", interpolation='none')
        # plt.colorbar()
    
    if filename:
        plt.savefig(filename)
    plt.close()
    
def load_config(cfg):
    with open(cfg, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_log_dir(cfg):
    log_dir = cfg["log_dir"] + '_' + cfg["model"]["name"]
    # if cfg["stage"] == 0:
    #     log_dir = os.path.join(log_dir, "stage0")
    print(f"Logs and models are saved at '{log_dir}'.")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def prepare_data(datadir):
    """prepare data list"""
    #get 100 random values between 0 and 2000
    images = sorted(glob(os.path.join(datadir, "**/*img.jpg"), recursive = True))
    labels = sorted(glob(os.path.join(datadir, "**/*mask.jpg"), recursive = True))
    print('Number of images:', len(images))
    data_list = [
        {"img": _image, "label": _label, 'case_class': os.path.dirname(_image).split('/')[-3], 'case_id': os.path.dirname(_image).split('/')[-2]}
        for _image, _label in zip(images, labels)
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
    if cfg['model']['name'] == 'Unet' or cfg['model']['name'] == 'UNETR' or cfg['model']['name'] == 'SwinUNETR':
        
        if cfg['preprocessing']['image_preprocess'] == 'CropPosNeg':
            if phase == 'train':
                train_transforms = Compose(
                        [
                        LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                        EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                        EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                        SplitLabelMined(keys="label"),
                        RandCropByPosNegLabeld(
                            keys=["img", "label"], label_key="label", spatial_size= cfg['preprocessing']['roi_size'], pos=3, neg=1, num_samples=cfg['preprocessing']['num_samples_per_image']
                        ),
                        # RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=0),
                        # RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=1),
                        # RandRotate90d(keys=("img", "label"), prob=0.5, spatial_axes=(0, 1)),
                        OneOf(
                            transforms=[
                                RandGaussianSmoothd(keys=["img"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
                                MedianSmoothd(keys=["img"], radius=1),
                                RandGaussianNoised(keys=["img"], prob=1.0, std=0.05),
                            ]
                        ),   
                        ScaleIntensityRangeD(keys=("img"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0, clip=True), 
                        SelectItemsd(keys=("img", "label","case_id", "case_class")),
                        ]
                    )
                return train_transforms
            elif phase == 'val' or phase == 'test':
                val_transforms = Compose(
                        [
                        LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                        EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                        EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                        SplitLabelMined(keys="label"),
                        ScaleIntensityRangeD(keys=("img"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0, clip=True),
                        SelectItemsd(keys=("img", "label","case_id", "case_class")),
                        ]
                    )
                return val_transforms
            else:
                raise ValueError(f"Unknown phase: {phase}")
            
        elif cfg['preprocessing']['image_preprocess'] == 'Resize':
            if phase == 'train':
                train_transforms = Compose(
                        [
                            LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                            EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                            EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                            SplitLabelMined(keys="label"),
                            Resized(keys=("img", "label"), spatial_size=cfg['preprocessing']['roi_size']),
                            # RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=0),
                            # RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=1),
                            # RandRotate90d(keys=("img", "label"), prob=0.5, spatial_axes=(0, 1)),
                            OneOf(
                                transforms=[
                                    RandGaussianSmoothd(keys=["img"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
                                    MedianSmoothd(keys=["img"], radius=1),
                                    RandGaussianNoised(keys=["img"], prob=1.0, std=0.05),
                                ]
                            ), 
                            ScaleIntensityRangeD(keys=("img"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0, clip=True),
                            SelectItemsd(keys=("img", "label","case_id", "case_class")),
                            ]
                        )
                return train_transforms
            elif phase == 'val' or phase == 'test':
                val_transforms = Compose(
                        [
                            LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                            EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                            EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                            SplitLabelMined(keys="label"),
                            Resized(keys=("img", "label"), spatial_size=cfg['preprocessing']['roi_size']),
                            ScaleIntensityRangeD(keys=("img"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0, clip=True),
                            SelectItemsd(keys=("img", "label","case_id", "case_class")),
                            ]
                        )
                return val_transforms
            else:
                raise ValueError(f"Unknown phase: {phase}")
            
        else:
            raise ValueError(f"Unknown image_preprocess: {cfg['preprocessing']['image_preprocess']}")
    elif cfg['model']['name'] == 'HoverNet':
        if phase == 'train':
            train_transforms = Compose(
                    [
                        LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                        EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                        EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                        SplitLabelMined(keys="label"),
                        # RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=0),
                        # RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=1),
                        # RandRotate90d(keys=("img", "label"), prob=0.5, spatial_axes=(0, 1)),
                        OneOf(
                            transforms=[
                                RandGaussianSmoothd(keys=["image"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
                                MedianSmoothd(keys=["image"], radius=1),
                                RandGaussianNoised(keys=["image"], prob=1.0, std=0.05),
                            ]
                        ),
                        #compute HoVer maps if model name is HoVer-Net
                        ComputeHoVerMaps(keys=["label"]),
                        #compute 
                        ScaleIntensityRangeD(keys=("img"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0, clip=True),
                        SelectItemsd(keys=("img", "label", "hover_label", "case_id", "case_class")),
                        ]
                    )
            return train_transforms
        elif phase == 'val' or phase == 'test':
            val_transforms = Compose(
                    [
                        LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                        EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                        EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                        SplitLabelMined(keys="label"),
                        ScaleIntensityRangeD(keys=("img"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0, clip=True),
                        SelectItemsd(keys=("img", "label", "case_id", "case_class")),
                        ]
                    )
            return val_transforms
        else:
            raise ValueError(f"Unknown phase: {phase}")
    else:
        raise ValueError(f"Unknown model name: {cfg['model']['name']}")
    
    
def get_model(cfg, pretrained_path = None):
    device = torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu")
    # create UNet, DiceLoss and Adam optimizer
    if cfg['model']['name'] == "Unet":
        model = monai.networks.nets.UNet(
            spatial_dims=cfg['model']['params']['spatial_dims'],
            in_channels=cfg['model']['params']['in_channels'],
            out_channels=cfg['model']['params']['out_channels'],
            channels=cfg['model']['params']['f_maps_channels'],
            strides=cfg['model']['params']['strides'],
            num_res_units=cfg['model']['params']['num_res_units'],
        ).to(device)
    elif cfg['model']['name'] == "UNETR":
        model = monai.networks.nets.UNETR(
            in_channels=cfg['model']['params']['in_channels'],
            out_channels=cfg['model']['params']['out_channels'],
            img_size=cfg['preprocessing']['roi_size'],
            spatial_dims = cfg['model']['params']['spatial_dims'],
            feature_size=cfg['model']['params']['feature_size'],
            hidden_size=cfg['model']['params']['hidden_size'],
            mlp_dim=cfg['model']['params']['mlp_dim'],
            num_heads=cfg['model']['params']['num_heads'],
        ).to(device)
    elif cfg['model']['name'] == "SwinUNETR":
        model = monai.networks.nets.SwinUNETR(
            img_size=cfg['preprocessing']['roi_size'],
            in_channels=cfg['model']['params']['in_channels'],
            out_channels=cfg['model']['params']['out_channels'],
            spatial_dims = cfg['model']['params']['spatial_dims'],
            depths = cfg['model']['params']['depths'],
            num_heads = cfg['model']['params']['num_heads'],
            feature_size = cfg['model']['params']['feature_size'],
            use_v2 = cfg['model']['params']['use_v2'],
        ).to(device)
    elif cfg['model']['name'] == "HoVerSwinUNETR":
        model = monai.networks.nets.HoVerSwinUNETR(
            img_size=cfg['preprocessing']['roi_size'],
            in_channels=cfg['model']['params']['in_channels'],
            out_channels=cfg['model']['params']['out_channels'],
            spatial_dims = cfg['model']['params']['spatial_dims'],
            depths = cfg['model']['params']['depths'],
            num_heads = cfg['model']['params']['num_heads'],
            feature_size = cfg['model']['params']['feature_size'],
            hovermaps = cfg['model']['params']['hovermaps'],
            freeze_encoder = cfg['model']['params']['freeze_encoder'],
            freeze_decoder_bin = cfg['model']['params']['freeze_decoder_bin'],
        ).to(device)
    else:
        raise ValueError(f"Model {cfg['model']['name']} not implemented")
    
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
        print(f"Model loaded from {pretrained_path}")
    
    model.eval()
    
    return model
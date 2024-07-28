# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import yaml
import logging
import torch
import numpy as np
import cv2
import torch.distributed as dist
from argparse import ArgumentParser
from tqdm import tqdm
import wandb
import random
from typing import Optional, Dict

import torch.nn as nn
import torch.nn.functional as F

from cellseg_models_pytorch.models import HoverNet, CellVitSAM
from cellseg_models_pytorch.losses import JointLoss, MultiTaskLoss, CELoss, DiceLoss

from monai.data import DataLoader, partition_dataset, CacheDataset, decollate_batch
# from monai.networks.nets import HoVerNet
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.transforms import (
    MapTransform,
    LoadImaged,
    EnsureChannelFirstd,
    TorchVisiond,
    Lambdad,
    Activationsd,
    OneOf,
    MedianSmoothd,
    AsDiscreted,
    Compose,
    CastToTyped,
    ComputeHoVerMapsd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandFlipd,
    RandAffined,
    RandGaussianSmoothd,
    CenterSpatialCropd,
    RandCropByPosNegLabeld,
    CopyItemsd,
    Compose,
    AsChannelLastd,
)
# from monai.handlers import (
#     MeanDice,
#     CheckpointSaver,
#     LrScheduleHandler,
#     StatsHandler,
#     TensorBoardStatsHandler,
#     ValidationHandler,
#     from_engine,
# )
from monai.utils import set_determinism
# from monai.utils.enums import HoVerNetBranch
# from monai.inferers import sliding_window_inference
# from monai.apps.pathology.handlers.utils import from_engine_hovernet
# from monai.apps.pathology.engines.utils import PrepareBatchHoVerNet
# from monai.apps.pathology.losses import HoVerNetLoss
from monai.apps.pathology.inferers import SlidingWindowHoVerNetInferer
from monai.optimizers.lr_scheduler import WarmupCosineSchedule

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from skimage import color, measure


class Dict2Class(object):
    # ToDo: Wrap into RandStainNA
    def __init__(self, my_dict: Dict):
        self.my_dict = my_dict
        for key in my_dict:
            setattr(self, key, my_dict[key])


def get_yaml_data(yaml_file):
    # ToDo: Wrap into RandStainNA
    file = open(yaml_file, "r", encoding="utf-8")
    file_data = file.read()
    file.close()
    # str->dict
    data = yaml.load(file_data, Loader=yaml.FullLoader)

    return data


class RandStainNA(object):
    # ToDo: support downloading yaml file from online if the path is not provided.
    def __init__(
        self,
        yaml_file: str,
        std_hyper: Optional[float] = 0,
        distribution: Optional[str] = "normal",
        probability: Optional[float] = 1.0,
        is_train: Optional[bool] = True,
    ):

        # true:training setting/false: demo setting

        assert distribution in [
            "normal",
            "laplace",
            "uniform",
        ], "Unsupported distribution style {}.".format(distribution)

        self.yaml_file = yaml_file
        cfg = get_yaml_data(self.yaml_file)
        c_s = cfg["color_space"]

        self._channel_avgs = {
            "avg": [
                cfg[c_s[0]]["avg"]["mean"],
                cfg[c_s[1]]["avg"]["mean"],
                cfg[c_s[2]]["avg"]["mean"],
            ],
            "std": [
                cfg[c_s[0]]["avg"]["std"],
                cfg[c_s[1]]["avg"]["std"],
                cfg[c_s[2]]["avg"]["std"],
            ],
        }
        self._channel_stds = {
            "avg": [
                cfg[c_s[0]]["std"]["mean"],
                cfg[c_s[1]]["std"]["mean"],
                cfg[c_s[2]]["std"]["mean"],
            ],
            "std": [
                cfg[c_s[0]]["std"]["std"],
                cfg[c_s[1]]["std"]["std"],
                cfg[c_s[2]]["std"]["std"],
            ],
        }

        self.channel_avgs = Dict2Class(self._channel_avgs)
        self.channel_stds = Dict2Class(self._channel_stds)

        self.color_space = cfg["color_space"]
        self.p = probability
        self.std_adjust = std_hyper
        self.color_space = c_s
        self.distribution = distribution
        self.is_train = is_train

    def _getavgstd(self, image: np.ndarray, isReturnNumpy: Optional[bool] = True):

        avgs = []
        stds = []

        num_of_channel = image.shape[2]
        for idx in range(num_of_channel):
            avgs.append(np.mean(image[:, :, idx]))
            stds.append(np.std(image[:, :, idx]))

        if isReturnNumpy:
            return (np.array(avgs), np.array(stds))
        else:
            return (avgs, stds)

    def _normalize(
        self,
        img: np.ndarray,
        img_avgs: np.ndarray,
        img_stds: np.ndarray,
        tar_avgs: np.ndarray,
        tar_stds: np.ndarray,
    ) -> np.ndarray:

        img_stds = np.clip(img_stds, 0.0001, 255)
        img = (img - img_avgs) * (tar_stds / img_stds) + tar_avgs

        if self.color_space in ["LAB", "HSV"]:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def augment(self, img):
        # img:is_train:false——>np.array()(cv2.imread()) #BGR
        # img:is_train:True——>PIL.Image #RGB

        if self.is_train == False:
            image = img
        else:
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        num_of_channel = image.shape[2]

        # color space transfer
        if self.color_space == "LAB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.color_space == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == "HED":
            image = color.rgb2hed(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        std_adjust = self.std_adjust

        # virtual template generation
        tar_avgs = []
        tar_stds = []
        if self.distribution == "uniform":

            # three-sigma rule for uniform distribution
            for idx in range(num_of_channel):

                tar_avg = np.random.uniform(
                    low=self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx],
                    high=self.channel_avgs.avg[idx] + 3 * self.channel_avgs.std[idx],
                )
                tar_std = np.random.uniform(
                    low=self.channel_stds.avg[idx] - 3 * self.channel_stds.std[idx],
                    high=self.channel_stds.avg[idx] + 3 * self.channel_stds.std[idx],
                )

                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)
        else:
            if self.distribution == "normal":
                np_distribution = np.random.normal
            elif self.distribution == "laplace":
                np_distribution = np.random.laplace

            for idx in range(num_of_channel):
                tar_avg = np_distribution(
                    loc=self.channel_avgs.avg[idx],
                    scale=self.channel_avgs.std[idx] * (1 + std_adjust),
                )

                tar_std = np_distribution(
                    loc=self.channel_stds.avg[idx],
                    scale=self.channel_stds.std[idx] * (1 + std_adjust),
                )
                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)

        tar_avgs = np.array(tar_avgs)
        tar_stds = np.array(tar_stds)

        img_avgs, img_stds = self._getavgstd(image)

        image = self._normalize(
            img=image,
            img_avgs=img_avgs,
            img_stds=img_stds,
            tar_avgs=tar_avgs,
            tar_stds=tar_stds,
        )

        if self.color_space == "LAB":
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        elif self.color_space == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.color_space == "HED":
            nimg = color.hed2rgb(image)
            imin = nimg.min()
            imax = nimg.max()
            rsimg = (255 * (nimg - imin) / (imax - imin)).astype(
                "uint8"
            )  # rescale to [0,255]

            image = cv2.cvtColor(rsimg, cv2.COLOR_RGB2BGR)

        # return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __call__(self, img):
        if np.random.rand(1) < self.p:
            return self.augment(img)
        else:
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += f"methods=Reinhard"
        format_string += f", colorspace={self.color_space}"
        format_string += f", mean={self._channel_avgs}"
        format_string += f", std={self._channel_stds}"
        format_string += f", std_adjust={self.std_adjust}"
        format_string += f", distribution={self.distribution}"
        format_string += f", p={self.p})"
        return format_string


class RandStainNAd(MapTransform):
    """
    Convert RandStainNA to a MONAI dict-based transform.
    """
    def __init__(
        self,
        keys,
        yaml_file: str,
        std_hyper: Optional[float] = 0,
        distribution: Optional[str] = "normal",
        probability: Optional[float] = 1.0,
        is_train: Optional[bool] = True,
    ):
        super().__init__(keys)
        self.rand_stain_na_transform = RandStainNA(
            yaml_file=yaml_file,
            std_hyper=std_hyper,
            distribution=distribution,
            probability=probability,
            is_train=is_train,
        )

    def __call__(self, data):
        # Copy the input data to avoid changing the original data
        d = dict(data)
        # Apply the transformation only to the specified keys
        for key in self.keys:
            d[key] = self.rand_stain_na_transform(d[key])
        return d



def create_log_dir(cfg):

    log_dir = os.path.join(cfg["log_dir"], cfg["exp_name"], f'stage_{cfg["stage"]}')
    print(f"Logs and models are saved at '{log_dir}'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    return log_dir



def prepare_data(data_dir, fold_num=None):

    if fold_num is not None:
        data_dir = os.path.join(data_dir, f'fold_{fold_num}')

    images = sorted(glob.glob(os.path.join(data_dir, '**/*img.jpg'), recursive = True))
    masks = sorted(glob.glob(os.path.join(data_dir, '**/*mask.jpg'), recursive = True))
    print(f'{len(images)} images and {len(masks)} masks found in {data_dir}')

    return [{"image": _img, "label": _mask} for _img, _mask in zip(images, masks)]



def get_loaders(cfg, train_transforms, val_transforms):
    multi_gpu = True if torch.cuda.device_count() > 1 else False

    # Validation data
    if cfg.get("val_fold"):
        val_data = prepare_data(cfg["data_path"], cfg["val_fold"])
    else:
        val_data = prepare_data(cfg["val_path"])

    # Training data
    if cfg.get("n_folds"):
        train_data = []
        for fold in range(cfg['nfolds']):
            if cfg.get('val_fold') and fold == cfg['val_fold']:
                continue
            train_data.extend(prepare_data(cfg["data_path"], fold))
    else:
        train_data = prepare_data(cfg["data_path"])

    print("train_files:", len(train_data))
    print("val_files:", len(val_data))

    if multi_gpu:
        train_data = partition_dataset(
            data=train_data,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
            shuffle=True,
            seed=cfg["seed"],
        )[dist.get_rank()]
        val_data = partition_dataset(
            data=val_data,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
            shuffle=False,
            seed=cfg["seed"],
        )[dist.get_rank()]

    train_ds = CacheDataset(
        data=train_data,
        transform=train_transforms, 
        cache_rate=cfg["cache_rate"], 
        num_workers=4
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_ds = CacheDataset(
        data=val_data,
        transform=val_transforms, 
        cache_rate=cfg["cache_rate"], 
        num_workers=4
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg["batch_size"], 
        num_workers=cfg["num_workers"],
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader



def create_model(cfg, device):

    model_class = HoverNet if cfg["model_name"].lower() == "hovernet" else CellVitSAM

    # Stage 0 - Load pre-trained SAM ViT encoder model [Freeze encoder: True]
    if cfg["stage"] == 0:
        model = model_class(
            decoders=('hovernet', 'label'),
            heads={
                'hovernet': {'hovernet' : 2},
                'label': {'label': 2},
            },
            enc_name=cfg["enc_name"],
            enc_pretrain=True,
            enc_freeze=True
        ).to(device)
        print(f'stage{cfg["stage"]} start!')

    # Stage 1 - Load pre-trained model from checkpoint [Freeze encoder: False]
    elif cfg["stage"] == 1:
        model = model_class(
            decoders=('hovernet', 'label'),
            heads={
                'hovernet': {'hovernet' : 2},
                'label': {'label': 2},
            },
            enc_name=cfg['enc_name'], #resnet50
            enc_pretrain=False,
            enc_freeze=False
        ).to(device)
        checkpoint = torch.load(cfg["ckpt"])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'stage{cfg["stage"]}, success load weight!')

    # Stage 2 - Load pre-trained SAM ViT encoder model [Freeze encoder: False]
    elif cfg["stage"] == 2:
        model = model_class(
            decoders=('hovernet', 'label'),
            heads={
                'hovernet': {'hovernet' : 2},
                'label': {'label': 2},
            },
            enc_name=cfg["enc_name"],
            enc_pretrain=True,
            enc_freeze=False
        ).to(device)
        print(f'stage{cfg["stage"]} start!')

    return model


def msge_loss(true, pred, focus):
    """Calculate the mean squared error of the gradients of 
    horizontal and vertical map predictions. Assumes 
    channel 0 is Vertical and channel 1 is Horizontal.

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps 
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)
    
    Returns:
        loss:  mean squared error of gradients

    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    ####
    def get_gradient_hv(hv):
        """For calculating gradient."""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[:, 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[:, 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss


# Quick wrapper for MSE loss to make it fit the JointLoss API
class MSELoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(
        self, yhat: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return F.mse_loss(yhat, target, reduction="mean")


# Quick wrapper for MSGE loss to make it fit the JointLoss API
class MSGELoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(
        self, yhat: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return msge_loss(yhat, target, focus=mask)



def run(cfg):

    set_determinism(seed=cfg["seed"])

    # cfg["out_size"] = [512, 512]

    multi_gpu = True if torch.cuda.device_count() > 1 else False
    if multi_gpu:
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda:{}".format(dist.get_rank()))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if cfg["use_gpu"] else "cpu")

    # --------------------------------------------------------------------------
    # Data Loading and Preprocessing
    # --------------------------------------------------------------------------
    # __________________________________________________________________________
    # Build MONAI preprocessing
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], dtype=torch.uint8),
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg["patch_size"],
                pos=9,
                neg=1,
                num_samples=1,
            ),
            AsChannelLastd(keys=["image"], channel_dim=0),
            RandStainNAd(
                keys=["image"],
                yaml_file=cfg['randStainNA']['yaml_file'], 
                std_hyper=cfg['randStainNA']['std_hyper'],
                distribution=cfg['randStainNA']['distribution'], 
                probability=cfg['randStainNA']['probability'], 
                is_train=True
            ),
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            Lambdad(
                keys="label", 
                func=lambda x: measure.label(x), 
                overwrite="inst_label"
            ),
            # RandAffined(
            #     keys=["image", "inst_label"],
            #     prob=1.0,
            #     rotate_range=((np.pi), 0),
            #     scale_range=((0.2), (0.2)),
            #     shear_range=((0.05), (0.05)),
            #     translate_range=((6), (6)),
            #     padding_mode="zeros",
            #     mode=("nearest"),
            # ),
            RandFlipd(keys=["image", "inst_label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "inst_label"], prob=0.5, spatial_axis=1),
            # OneOf( # TODO: Check if this is correct
            #     transforms=[
            #         RandGaussianSmoothd(keys=["image"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
            #         MedianSmoothd(keys=["image"], radius=1),
            #         RandGaussianNoised(keys=["image"], prob=1.0, std=0.05),
            #     ]
            # ),
            # CastToTyped(keys=["image", "inst_label"], dtype=torch.int),
            CastToTyped(keys=["inst_label"], dtype=torch.int),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            # TorchVisiond( # TODO: Check if this is correct
            #     keys=["image"],
            #     name="ColorJitter",
            #     brightness=0.1,
            #     contrast=(0.95, 1.10),
            #     saturation=0.2,
            #     hue=0.04
            # ),
            ComputeHoVerMapsd(keys="inst_label", new_key_prefix="hovernet_"),
            # CopyItemsd(keys=["hover_inst_label"], names=["hovernet"]),
            Lambdad(keys="inst_label", func=lambda x: x > 0, overwrite="label"),
            # CenterSpatialCropd(
            #     keys=["label", "hover_inst_label", "inst_label"],
            #     roi_size=cfg["out_size"],
            # ),
            # AsDiscreted(keys=["label"], to_onehot=2),
            AsDiscreted(keys=["label"], threshold=0.5),
            CastToTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], dtype=torch.uint8),
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
            Lambdad(
                keys="label", 
                func=lambda x: measure.label(x), 
                overwrite="inst_label"
            ),
            # CastToTyped(keys=["image", "inst_label"], dtype=torch.int),
            CastToTyped(keys=["inst_label"], dtype=torch.int),
            
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            ComputeHoVerMapsd(keys="inst_label", new_key_prefix="hovernet_"),
            # CopyItemsd(keys=["hover_inst_label"], names=["hovernet"]),
            Lambdad(keys="inst_label", func=lambda x: x > 0, overwrite="label"),
            # CenterSpatialCropd(
            #     keys=["image", "hovernet_inst_label", "label"],
            #     roi_size=cfg["patch_size"],
            # ),
            RandCropByPosNegLabeld(
                keys=["image", "label", "hovernet_inst_label"],
                label_key="label",
                spatial_size=cfg["patch_size"],
                pos=1,
                neg=0,
                num_samples=1,
            ),
            AsDiscreted(keys=["label"], threshold=0.5),
            CastToTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )

    # __________________________________________________________________________
    # Create MONAI DataLoaders
    train_loader, val_loader = get_loaders(cfg, train_transforms, val_transforms)

    # --------------------------------------------------------------------------
    # Create Model, Loss, Optimizer, lr_scheduler
    # --------------------------------------------------------------------------
    # __________________________________________________________________________
    # initialize model
    model = create_model(cfg, device)
    if multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist.get_rank()], output_device=dist.get_rank()
        )

    # loss_function = HoVerNetLoss(lambda_hv_mse=1.0)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=cfg["lr"], weight_decay=1e-5)
    
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_step"])

    lr_scheduler = WarmupCosineSchedule(
        optimizer, 
        warmup_steps=cfg['scheduler']['warmup_epochs'] * len(train_loader), 
        t_total=cfg['epochs'] * len(train_loader), 
        cycles = cfg['scheduler']['cycles'], 
        end_lr=1e-9
    )

    # post_process_np = Compose(
    #     [
    #         Activationsd(keys=HoVerNetBranch.NP.value, softmax=True),
    #         AsDiscreted(keys=HoVerNetBranch.NP.value, argmax=True),
    #     ]
    # )
    # post_process = Lambdad(keys="pred", func=post_process_np)
    post_process = Compose([
        Activationsd(keys='label', softmax=True), 
        AsDiscreted(keys='label', argmax=True, dim=1)
    ])

    # # --------------------------------------------
    # # Ignite Trainer/Evaluator
    # # --------------------------------------------
    # # Evaluator
    # val_handlers = [
    #     CheckpointSaver(
    #         save_dir=cfg["log_dir"],
    #         save_dict={"model": model},
    #         save_key_metric=True,
    #     ),
    #     StatsHandler(output_transform=lambda x: None),
    #     TensorBoardStatsHandler(log_dir=cfg["log_dir"], output_transform=lambda x: None),
    # ]
    # if multi_gpu:
    #     val_handlers = val_handlers if dist.get_rank() == 0 else None
    # evaluator = SupervisedEvaluator(
    #     device=device,
    #     val_data_loader=val_loader,
    #     prepare_batch=PrepareBatchHoVerNet(extra_keys=["label_type", "hover_inst_label"]),
    #     network=model,
    #     postprocessing=post_process,
    #     key_val_metric={
    #         "val_dice": MeanDice(
    #             include_background=False,
    #             output_transform=from_engine_hovernet(keys=["pred", "label"], nested_key=HoVerNetBranch.NP.value),
    #         )
    #     },
    #     val_handlers=val_handlers,
    #     amp=cfg["amp"],
    # )

    # # Trainer
    # train_handlers = [
    #     LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
    #     ValidationHandler(validator=evaluator, interval=cfg["val_freq"], epoch_level=True),
    #     CheckpointSaver(
    #         save_dir=cfg["log_dir"],
    #         save_dict={"model": model, "opt": optimizer},
    #         save_interval=cfg["save_interval"],
    #         save_final=True,
    #         final_filename="model.pt",
    #         epoch_level=True,
    #     ),
    #     StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    #     TensorBoardStatsHandler(
    #         log_dir=cfg["log_dir"], tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
    #     ),
    # ]
    # if multi_gpu:
    #     train_handlers = train_handlers if dist.get_rank() == 0 else train_handlers[:2]
    
    # # trainer = SupervisedTrainer(
    # #     device=device,
    # #     max_epochs=cfg['epochs'],
    # #     train_data_loader=train_loader,
    # #     prepare_batch=PrepareBatchHoVerNet(extra_keys=["label_type", "hover_label_inst"]),
    # #     network=model,
    # #     optimizer=optimizer,
    # #     loss_function=loss_function,
    # #     postprocessing=post_process,
    # #     key_train_metric={
    # #         "train_dice": MeanDice(
    # #             include_background=False,
    # #             output_transform=from_engine_hovernet(keys=["pred", "label"], nested_key=HoVerNetBranch.NP.value),
    # #         )
    # #     },
    # #     train_handlers=train_handlers,
    # #     amp=cfg["amp"],
    # # )
    # # trainer.run()

    # Define the multi-task loss function
    hovernet_loss_dict = {
        "hovernet": JointLoss([MSELoss(), MSGELoss()]) #, weights=[0.5, 1.0]),
    }
    hovernet_branch_loss = MultiTaskLoss(branch_losses=hovernet_loss_dict)

    # semantic_loss_dict = {
    #     "label": JointLoss([CELoss(apply_sd=True), DiceLoss()]) #,weights=[0.5, 0.5]),
    # }
    # semantic_branch_loss = MultiTaskLoss(branch_losses=semantic_loss_dict)
    semantic_branch_loss = DiceCELoss(include_background=False, softmax=True,
                                      to_onehot_y=True)

    sliding_inferer = SlidingWindowHoVerNetInferer(roi_size=cfg["patch_size"], 
                                                   sw_batch_size=1, 
                                                   overlap=0.5,)
    val_metric = DiceMetric(
        include_background=False, 
        reduction="mean", 
    )

    wandb.init(
        job_type='training',
        project=cfg['wandb']['project'], 
        entity = cfg['wandb']['entity'],
        name=cfg['exp_name'],
        save_code=True, 
        reinit=cfg['wandb']['reinit'], 
        resume=cfg['wandb']['resume'],
        mode=cfg['wandb']['mode'],
        config = cfg
    )

    first_epoch = 0
    best_val_metric = -1

    for epoch in range(first_epoch, cfg['epochs']):
        
        model.train()
        epoch_loss = 0
        step=0

        print('\n\n')

        with tqdm(train_loader, unit="batch") as loader:
            loader.set_description(f"Epoch {epoch:03}")

            for batch in loader:

                step += 1

                # run forward pass and compute loss
                soft_masks = model(batch["image"].to(device))
                targets = {k.split("_")[0]: v.to(device) \
                           for k, v in batch.items() if k != "image"}

                hovernet_loss = hovernet_branch_loss(soft_masks, targets,
                                                     mask=targets["label"])
                semantic_loss = semantic_branch_loss(soft_masks['label'], 
                                                     targets['label'])
                loss = hovernet_loss + semantic_loss
                epoch_loss += loss.detach().float()

                loader.set_postfix_str(f"epoch_loss: {epoch_loss.item()/step:.4f}, " \
                                       f"batch_loss: {loss.item():.4f}, " \
                                       f"hovernet_loss: {hovernet_loss.item():.4f}, " \
                                       f"semantic_loss: {semantic_loss.item():.4f}, " \
                                       f"lr: {lr_scheduler.get_last_lr()[0]:.8f}")

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        wandb.log(
            {
                "train/loss": epoch_loss.item() / step,
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "epoch": epoch + 1
            }
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            #save rng state of torch, np and random
            'torch_rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate()
        }, os.path.join(cfg["log_dir"], "last_model.pth"))

        # Save random state
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()

        # Validation
        if (epoch + 1) % cfg['val_freq'] == 0:
            validation_dice = 0
            model.eval()

            # Seed fixed for validation
            set_determinism(seed=cfg["seed"])

            with torch.no_grad():
                with tqdm(val_loader, unit="batch") as loader:
                    loader.set_description(f"Validation Epoch {epoch:03}")

                    for batch in loader:

                        val_outputs = model(batch["image"].to(device))
                        # val_outputs = sliding_inferer(
                        #     inputs=batch["image"].to(device), 
                        #     network=model
                        # )
                        
                        val_outputs = post_process(val_outputs)
                        val_metric(y_pred=val_outputs['label'], y=batch['label'].to(device))

                    validation_dice = val_metric.aggregate().item()
                    loader.set_postfix_str(f"val_dice: {round(validation_dice, 4)}")

            wandb.log(
                {
                    "val/dice_metric": validation_dice,
                    "epoch": epoch + 1
                }
            )
            
            val_metric.reset()

            if validation_dice > best_val_metric:
                best_val_metric = validation_dice
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        #save rng state of torch, np and random
                        'torch_rng_state': torch.get_rng_state(),
                        'np_rng_state': np.random.get_state(),
                        'random_rng_state': random.getstate()                   
                    }, 
                    os.path.join(cfg["log_dir"], "best_model.pth")
                )

                print("Best model saved!")

        # Restore training random state
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        torch.set_rng_state(torch_random_state)

    wandb.finish()

    if multi_gpu:
        dist.destroy_process_group()


def main():
    parser = ArgumentParser(description="Tumor detection on whole slide pathology images.")

    parser.add_argument("--config", help="configuration file", required=True)

    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default="/workspace/Data/Pathology/CoNSeP/Prepared",
    #     help="root data dir",
    # )
    # parser.add_argument("--log-dir", type=str, default="./logs/", help="log directory")
    # parser.add_argument("--exp_name", type=str, default="hovernet", help="experiment name")
    # parser.add_argument("--nfolds", type=int, default=3, help="number of folds")
    # parser.add_argument("--val_fold", type=int, default=0, help="validation fold")
    # parser.add_argument("-s", "--seed", type=int, default=24)

    # parser.add_argument("--bs", type=int, default=16, dest="batch_size", help="batch size")
    # parser.add_argument("--ep", type=int, default=50, dest="epochs", help="number of epochs")
    # parser.add_argument("--lr", type=float, default=1e-4, dest="lr", help="initial learning rate")
    # parser.add_argument("--lr_step", type=int, default=25, dest="lr_step", help="period of learning rate decay")
    # parser.add_argument("-f", "--val_freq", type=int, default=1, help="validation frequence")
    # parser.add_argument("--stage", type=int, default=0, help="training stage")
    # parser.add_argument("--no-amp", action="store_false", dest="amp", help="deactivate amp")
    # # parser.add_argument("--classes", type=int, default=5, dest="out_classes", help="output classes")
    # # parser.add_argument("--mode", type=str, default="fast", help="choose either `original` or `fast`")

    # parser.add_argument("--save_interval", type=int, default=10)
    # parser.add_argument("--num_workers", type=int, default=8, dest="num_workers", help="number of workers")
    # parser.add_argument("--no-gpu", action="store_false", dest="use_gpu", help="deactivate use of gpu")
    # parser.add_argument("--ckpt", type=str, dest="ckpt", help="model checkpoint path")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)

    logging.basicConfig(level=logging.INFO)
    cfg["log_dir"] = create_log_dir(cfg)

    # if cfg["stage"] == 1 and not cfg.get("ckpt") and cfg.get("log_dir"):
    #     cfg["ckpt"] = os.path.join(cfg["log_dir"], "stage_0", "last_model.pth")

    print(cfg)
    run(cfg)


if __name__ == "__main__":
    main()

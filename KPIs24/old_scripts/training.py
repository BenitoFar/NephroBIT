from abc import ABC, abstractmethod
import logging
import os
import sys
import tempfile
from glob import glob
import random
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import matplotlib.pyplot as plt

import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch, DataLoader, partition_dataset, CacheDataset
from monai.apps.nuclick.transforms import SplitLabeld
from monai.apps import CrossValidation
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    RandTorchVisiond,
    ScaleIntensityRangeD,
    SelectItemsd,
    OneOf,
    MedianSmoothd,
    AsDiscreted,
    CastToTyped,
    ComputeHoVerMapsd,
    RandGaussianNoised,
    RandFlipd,
    RandAffined,
    RandGaussianSmoothd,
    CenterSpatialCropd,
)
from monai.handlers import (
    MeanDice,
    CheckpointSaver,
    LrScheduleHandler,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.visualize import plot_2d_or_3d_image
from monai.utils import set_determinism
import wandb 

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class CVDataset(ABC, CacheDataset):
    """
    Base class to generate cross validation datasets.

    """

    def __init__(
        self,
        data,
        transform,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=4,
    ) -> None:
        data = self._split_datalist(datalist=data)
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    @abstractmethod
    def _split_datalist(self, datalist):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
    
    
def create_log_dir(cfg):
    log_dir = cfg["log_dir"]
    if cfg["stage"] == 0:
        log_dir = os.path.join(log_dir, "stage0")
    print(f"Logs and models are saved at '{log_dir}'.")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def prepare_data(datadir):
    """prepare data list"""
    images = sorted(glob(os.path.join(datadir, "**/*img.jpg"), recursive = True))[0:10]
    labels = sorted(glob(os.path.join(datadir, "**/*mask.jpg"), recursive = True))[0:10]
    
    data_list = [
        {"img": _image, "label": _label}
        for _image, _label in zip(images, labels)
    ]
    return data_list

def get_loaders(cfg, train_transforms, val_transforms):
    multi_gpu = True if torch.cuda.device_count() > 1 else False

    train_data = prepare_data(cfg["root"], "Train")
    valid_data = prepare_data(cfg["root"], "Validation")
    
    if multi_gpu:
        train_data = partition_dataset(
            data=train_data,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
            shuffle=True,
            seed=cfg["seed"],
        )[dist.get_rank()]
        valid_data = partition_dataset(
            data=valid_data,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
            shuffle=False,
            seed=cfg["seed"],
        )[dist.get_rank()]

    print("train_files:", len(train_data))
    print("val_files:", len(valid_data))

    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0, num_workers=4)
    valid_ds = CacheDataset(data=valid_data, transform=val_transforms, cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        valid_ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def train(index_fold, train_loaders, val_loaders, device):
    set_determinism(seed=0)
    seed_everything(0)
    
    #calculate lenght train_loaders[index_fold]
    train_ds = train_loaders[index_fold].dataset
    val_ds = val_loaders[index_fold].dataset
    print(f"train_ds length: {len(train_ds)}, val_ds length: {len(val_ds)}")
    
    #define metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # create UNet, DiceLoss and Adam optimizer
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    wandb.watch(model, log="all")

    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    
    for epoch in range(100):
        wandb_dict = {}  # wandb to log
        
        print("-" * 10)
        print(f"epoch {epoch + 1}/{100}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loaders[index_fold]:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loaders[index_fold].batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            print("train_loss", loss.item(), epoch_len * epoch + step)
            
            # Update wandb dict
            wandb_dict.update({
                "train_loss": loss,
            })
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loaders[index_fold]:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    roi_size = (3, 512, 512)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                # Update wandb dict
                wandb_dict.update({
                    "val_dice": dice_metric,
                })
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    wandb.save('model_best.pth')
    wandb.finish()
    
    
def show_image(image, label, points=None):
    print(f"Image: {image.shape}; Label: {label.shape}")

    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, 2)
        label = np.moveaxis(label, 0, 2)

    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(image, cmap="gray")

    if label is not None:
        masked = np.ma.masked_where(label == 0, label)
        plt.imshow(masked, "jet", interpolation="none", alpha=0.5)

    plt.colorbar()

    if label is not None:
        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(label)
        plt.colorbar()
        
def main(datadir, results_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    set_determinism(seed=0)
    seed_everything(0)
     
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_list = prepare_data(datadir)
    
    # define transforms for image and labelmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "label"]),
            EnsureChannelFirstd(keys=["img"], channel_dim=-1),
            EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
            SplitLabeld(keys="label", mask_value=None, others_value=255),
            RandCropByPosNegLabeld(
                keys=["img", "label"], label_key="label", spatial_size=[512, 512], pos=1, neg=1, num_samples=4
            ),
            RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=0),
            RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=1),
            RandRotate90d(keys=("img", "label"), prob=0.5, spatial_axes=(0, 1)),
            ScaleIntensityRangeD(keys="img", a_min=0.0, a_max=255.0, b_min=0, b_max=1.0),
                
            ]
        )
            
    val_transforms = Compose(
            [
                LoadImaged(keys=["img", "label"]),
                EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                SplitLabeld(keys="label", mask_value=None, others_value=255),
                ScaleIntensityRangeD(keys="img", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ]
    )


    # define dataset, data loader
    
    num_folds = 5
    folds = list(range(num_folds))

    cvdataset = CrossValidation(
        dataset_cls=CVDataset,
        data=data_list,
        nfolds=5,
        seed=333,
        transform=train_transforms,
    )

    train_dss = [cvdataset.get_dataset(folds=folds[0:i] + folds[(i + 1) :]) for i in folds]
    val_dss = [cvdataset.get_dataset(folds=i, transform=val_transforms) for i in range(num_folds)]

    train_loaders = [DataLoader(train_dss[i], batch_size=2, shuffle=True, num_workers=4) for i in folds]
    val_loaders = [DataLoader(val_dss[i], batch_size=1, num_workers=4) for i in folds]
    
    
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = train_loaders[0]
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["label"].shape)
    
    os.makedirs(os.path.join(results_dir, 'train_images'), exist_ok=True)
    for i in range(len(check_data["img"])):
        check_image, check_label = (check_data["img"][i], check_data["label"][i])
        print(f"image shape: {check_image.shape}, label shape: {check_label.shape}")
        show_image(check_image, check_label)
        plt.savefig(os.path.join(results_dir, 'train_images', f'train_sample{i}.png'))
        plt.show()
        # for j in range(len(check_image)):
        #     img = np.moveaxis(check_image[j]['img'], 0, -1)    
        #     #print min and max
        #     print(img.min(), img.max())
        #     label = np.moveaxis(check_label[j]['label'], 0, -1)    
            
    
    #train
    #define random groupname to identify all runs of the cross validation
    group_name = f"unet"
    
    for index_fold in range(num_folds):
        wandb.init(project="nefrobit", 
                   name=f"fold_{index_fold}", 
                   group=group_name,
                   entity = 'benitofarina5',
                   save_code=True, 
                   reinit=True, 
                   resume='allow'
                    )
        
        train(index_fold, train_loaders, val_loaders, device)
        
if __name__ == "__main__":
    
    # results_dir = '/mnt/atlas/nefrobit'
    results_dir = '/home/benito/script/KPIs24/results/'
    datadir = '/data/KPIs24/KPIs24 Training Data/Task1_patch_level/data/'
    main(datadir, results_dir)
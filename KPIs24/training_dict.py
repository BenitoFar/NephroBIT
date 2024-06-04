from abc import ABC, abstractmethod
import logging
import os
import sys
import numpy as np
import random
import time
import torch
import monai
from monai.data import decollate_batch, DataLoader, CacheDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.apps.nuclick.transforms import SplitLabelMined
from monai.apps import CrossValidation
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.utils.data import Subset
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityRangeD,
    RandFlipd,
    SelectItemsd,
    LabelToMaskd,
    Resized
)
import matplotlib.pyplot as plt
from monai.utils import set_determinism
import wandb 
import random
from utilites import seed_everything, prepare_data, load_config, show_image
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

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


def train(cfg, index_fold, train_loader, val_loader, device, results_dir):
    
    
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
    
    results_images_dir = os.path.join(results_dir, 'train_images')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_images_dir, exist_ok=True)
    progress = open(results_dir + '/progress_train.txt', 'w')
    #calculate lenght train_loaders[index_fold]
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    print(f"train_ds length: {len(train_ds)}, val_ds length: {len(val_ds)}")
    
    #define metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # create UNet, DiceLoss and Adam optimizer
    model = monai.networks.nets.UNet(
        spatial_dims=cfg['model']['params']['spatial_dims'],
        in_channels=cfg['model']['params']['in_channels'],
        out_channels=cfg['model']['params']['out_channels'],
        channels=cfg['model']['params']['f_maps_channels'],
        strides=cfg['model']['params']['strides'],
        num_res_units=cfg['model']['params']['num_res_units'],
    ).to(device)
    
    if cfg['wandb']['state']: wandb.watch(model, log="all")

    loss_function = monai.losses.DiceLoss(include_background=True, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), cfg['model']['optimizer']['params']['learning_rate'])
    
    if cfg['model']['scheduler']['name'] == "WarmupCosineSchedule":
        warmup_steps = int(cfg['model']['scheduler']['params']['warmup_epochs'] * len(train_ds) * cfg['preprocessing']['sw_batch_size'] / train_loader.batch_size)
        t_total = int(cfg['training']['epoch'] * len(train_ds) * cfg['preprocessing']['sw_batch_size'] / train_loader.batch_size)
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total, cycles = cfg['model']['scheduler']['params']['cycles'], end_lr=1e-9)
    elif cfg['model']['scheduler']['name'] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg['model']['scheduler']['params']['T_max'], eta_min=1e-9)
    else:
        raise ValueError(f"Scheduler {cfg['model']['scheduler']['name']} not implemented")
    
    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    
    epoch_start_time = time.time()
    
    for epoch in range(cfg['training']['epoch']):
        
        if cfg['wandb']['state']: wandb_dict = {}  # wandb to log
        
        print("-" * 10)
        print(f"epoch {epoch + 1}/{cfg['training']['epoch']}")
        model.train()
        epoch_loss = 0
        step = 0
        for idx_img, batch_data in enumerate(train_loader):

            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            #save image and label
            # if idx_img < 10:
            #     show_image(inputs[0].cpu().numpy(), labels[0].cpu().numpy(), None, os.path.join(results_images_dir, f'fold_{index_fold}_train_image_{idx_img}_{epoch}_{step}.png'))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            print("train_loss", loss.item(), " - ", epoch_len * epoch + step, "batches processed")
            
            # Update wandb dict
            if cfg['wandb']['state']:
                wandb.log({"train/loss": loss.item()})
                # wandb_dict.update({
                #     : loss.item(),
                # })
            progress.write(f'(epoch:{epoch+1} >> loss:{loss.item()}\n') 
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        
        print(f"epoch {epoch + 1} average train loss: {epoch_loss:.4f} - learning rate: {scheduler.get_lr()[0]} - time: {time.time() - epoch_start_time:.2f} seconds")

        # step scheduler after each epoch (cosine decay)
        scheduler.step()

        if cfg['wandb']['state']:
            # 🐝 log train_loss averaged over epoch to wandb
            # wandb_dict.update({"train/loss_epoch": epoch_loss})
            wandb.log({"train/loss_epoch": epoch_loss})
            wandb.log("epoch", epoch + 1)
            # 🐝 log learning rate after each epoch to wandb
            # wandb_dict.update({"learning_rate": scheduler.get_lr()[0]})
            wandb.log({"learning_rate": scheduler.get_lr()[0]})
        progress.write(f'(epoch:{epoch+1} >> train/loss_epoch:{epoch_loss}\n') 
        progress.write(f'(epoch:{epoch+1} >> learning_rate:{scheduler.get_lr()[0]}\n')
        
        if (epoch + 1) % cfg['training']['val_interval'] == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                
                for idx_val, val_data in enumerate(val_loader):
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    sw_batch_size = cfg['preprocessing']['sw_batch_size']
                    val_outputs = sliding_window_inference(val_images, cfg['preprocessing']['roi_size'], sw_batch_size, model)
                    #save output label image
                    # plt.figure()
                    # plt.imshow(torch.argmax(val_outputs, dim=0).cpu().numpy().squeeze(), cmap='gray')
                    # plt.savefig(os.path.join(results_images_dir,f'fold_{index_fold}_prediction_image_{idx_val}_{epoch}_{step}.png'))
                    if idx_val < 5:
                        show_image(val_images[0].cpu().numpy(), val_labels[0].cpu().numpy(),torch.argmax(val_outputs, dim=1), os.path.join(results_images_dir,f'fold_{index_fold}_prediction_image_{idx_val}_epoch_{epoch}.png'))
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # Update wandb dict
                if cfg['wandb']['state']:
                    # wandb_dict.update({"val/dice_metric": dice_metric})
                    wandb.log({"val/dice_metric": metric})
                    #save
                progress.write(f'(epoch:{epoch+1} >> val/dice_metric:{metric}\n')
                   
                
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(results_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current validation mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                                     
    print(f"train completed, best_metric DICE: {best_metric:.4f} at epoch: {best_metric_epoch}")
    if cfg['wandb']['state']: 
        wandb_dict.update({"best_dice_metric": best_metric, "best_metric_epoch": best_metric_epoch})
        wandb.log({"best_dice_metric": best_metric, "best_metric_epoch": best_metric_epoch})
        wandb.save('model_best.pth')
    progress.write(f'(train completed, best_metric DICE: {best_metric:.4f} at epoch: {best_metric_epoch}\n')
    
def main(cfg):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    cfg = load_config(cfg)
    
    # create_log_dir(cfg)
    
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
     
    device = torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu")
    
    datadir = cfg['datadir']
    results_dir = os.path.join(cfg['results_dir'], cfg['model']['name'], cfg['preprocessing']['image_preprocess'])
    
    data_list = prepare_data(datadir)
    
    #join id and class
    for i in range(len(data_list)):
        data_list[i]['img_stratify'] = data_list[i]['img_id'] + '_' + data_list[i]['img_class']
        
    #split data_list for cross validation - grouped stratify split based on type
    sgkf = StratifiedGroupKFold(n_splits=cfg['training']['nfolds'], shuffle=True, random_state=cfg['seed'])
    
    # define transforms for image and labelmentation
    if cfg['preprocessing']['image_preprocess'] == 'CropPosNeg':
        train_transforms = Compose(
                [
                LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                SplitLabelMined(keys="label"),
                RandCropByPosNegLabeld(
                    keys=["img", "label"], label_key="label", spatial_size= cfg['preprocessing']['roi_size'], pos=3, neg=1, num_samples=4
                ),
                RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=0),
                RandFlipd(keys=("img", "label"), prob=0.5, spatial_axis=1),
                RandRotate90d(keys=("img", "label"), prob=0.5, spatial_axes=(0, 1)),
                ScaleIntensityRangeD(keys=("img", "label"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0),    
                SelectItemsd(keys=("img", "label","img_id", "img_class")),
                ]
            )
        val_transforms = Compose(
                [
                LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                SplitLabelMined(keys="label"),
                ScaleIntensityRangeD(keys=("img", "label"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0),
                SelectItemsd(keys=("img", "label","img_id", "img_class")),
                ]
             )
    elif cfg['preprocessing']['image_preprocess'] == 'Resize':
        train_transforms = Compose(
            [
                LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                SplitLabelMined(keys="label"),
                Resized(keys=("img", "label"), spatial_size=cfg['preprocessing']['roi_size']),
                ScaleIntensityRangeD(keys=("img", "label"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0),
                SelectItemsd(keys=("img", "label","img_id", "img_class")),
                ]
            )
        val_transforms = Compose(
            [
                LoadImaged(keys=["img", "label"], dtype=torch.uint8),
                EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                SplitLabelMined(keys="label"),
                Resized(keys=("img", "label"), spatial_size=cfg['preprocessing']['roi_size']),
                ScaleIntensityRangeD(keys=("img", "label"), a_min=0.0, a_max=255.0, b_min=0, b_max=1.0),
                SelectItemsd(keys=("img", "label","img_id", "img_class")),
                ]
            )
    else:
        raise ValueError(f"Preprocessing {cfg['preprocessing']['image_preprocess']} not implemented")

    # define dataset, data loader
    folds = list(range(cfg['training']['nfolds']))

    # cvdataset = CrossValidation(
    #     dataset_cls=CVDataset,
    #     data=data_list,
    #     nfolds=cfg['training']['nfolds'],
    #     seed=cfg['seed'],
    #     transform=train_transforms,
    # )

    #train
    #define random groupname to identify all runs of the cross validation
    for index_fold, (train_index, val_index) in enumerate(sgkf.split(data_list['img'], data_list['img_class'], data_list['img_stratify'])):
        results_fold_dir = os.path.join(results_dir, f'fold_{index_fold}')
        
        train_dss = cvdataset.get_dataset(folds=folds[0:index_fold] + folds[(index_fold + 1) :])
        val_dss = cvdataset.get_dataset(folds=folds[index_fold], transform=val_transforms)
        train_loader = DataLoader(train_dss, batch_size=cfg['training']['train_batch_size'], shuffle=True, num_workers=cfg['training']['num_workers'], persistent_workers=True) 
        val_loader = DataLoader(val_dss, batch_size=cfg['training']['val_batch_size'], num_workers=cfg['training']['num_workers'], persistent_workers=True)

        # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
        check_loader = train_loader
        check_data = monai.utils.misc.first(check_loader)
        print(check_data["img"].shape, check_data["label"].shape)
        
        os.makedirs(os.path.join(results_fold_dir, f'train_images_examples1'), exist_ok=True)
        for i in range(len(check_data["img"])):
            check_image, check_label = (check_data["img"][i], check_data["label"][i])
            print(f"image shape: {check_image.shape}, label shape: {check_label.shape}")
            show_image(check_image, check_label, None, os.path.join(results_fold_dir, f'train_images_examples', f'train_sample_{i}.png'))
            
            
        if cfg['wandb']['state']:
            run_name = f"{cfg['wandb']['group_name']}_{cfg['model']['name']}-fold{index_fold:02}"
            wandb.init(project=cfg['wandb']['project'], 
                    name=run_name, 
                    group= f"{cfg['wandb']['group_name']}_{cfg['model']['name']}_5foldcv_{cfg['preprocessing']['image_preprocess']}",
                    entity = cfg['wandb']['entity'],
                    save_code=True, 
                    reinit=cfg['wandb']['reinit'], 
                    resume=cfg['wandb']['resume'],
                    config = cfg,
                        )
        
        train(cfg, index_fold, train_loader, val_loader, device, results_fold_dir)
        
        if cfg['wandb']['state']: wandb.finish()
        
if __name__ == "__main__":
    cfg = "config_train.yaml"
    main(cfg)

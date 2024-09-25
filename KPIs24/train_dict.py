from abc import ABC, abstractmethod
import logging
import os
import argparse
import sys
import monai.networks
import numpy as np
import random
import time
import torch
import monai
from monai.data import decollate_batch, DataLoader, CacheDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.optimizers import WarmupCosineSchedule
from monai.losses import DiceCELoss
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
    Resized,
    OneOf, 
    RandGaussianSmoothd, 
    MedianSmoothd, 
    RandGaussianNoised
)
from losses import HoVerNetLoss
from collections import Counter
import matplotlib.pyplot as plt
from monai.utils import set_determinism
import wandb 
import random
from utilites import seed_everything, prepare_data, load_config, save_mask, show_image, get_transforms, get_model
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from models import train
torch.autograd.set_detect_anomaly(True)

def main(cfg):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    cfg = load_config(cfg)
    
    if cfg['wandb']['state']:
        run_name = f"{cfg['wandb']['group_name']}_{cfg['model']['name'] + ('deep_supervision' if cfg['model']['name'] == 'DynUNet' and cfg['model']['params']['deep_supervision'] else '')}{('_'+ cfg['mode']['mode_type'] if len(cfg['mode']['mode_type']) > 1 else '')}-{('fold_' + str(cfg['val_fold']) if cfg['val_fold'] != 'validation_cohort' else 'validation_cohort')}"
        wandb.init(project=cfg['wandb']['project'], 
                name=run_name, 
                group= f"{cfg['wandb']['group_name']}_{cfg['model']['name'] + ('deep_supervision' if cfg['model']['name'] == 'DynUNet' and cfg['model']['params']['deep_supervision'] else '')}{('_'+ cfg['mode']['mode_type'] if len(cfg['mode']['mode_type']) > 1 else '')}_{cfg['nfolds']}foldcv_{cfg['preprocessing']['image_preprocess']}",
                entity = cfg['wandb']['entity'],
                save_code=True, 
                reinit=cfg['wandb']['reinit'], 
                resume=cfg['wandb']['resume'],
                config = cfg,
                    )
        
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
    
    datadir = cfg['datadir']
    
    #get validation list files
    if cfg['val_fold'] != 'validation_cohort':
        datadir_val = os.path.join(datadir, f"fold_{cfg['val_fold']}")
        data_list_val = prepare_data(datadir_val)
        
        #get train list files
        data_list_train = []
        for i in range(cfg['nfolds']):
            if i != cfg['val_fold']:
                datadir_train = os.path.join(datadir, f'fold_{i}')
                datalist_fold_train = prepare_data(datadir_train)
                data_list_train.extend(datalist_fold_train)
    else:
        #get train list files
        data_list_train = prepare_data(cfg['datadir'])
        data_list_val = prepare_data(cfg['datadir_val'])
    
    print('Number of training images:', len(data_list_train), 'Number of validation images:', len(data_list_val))
    train_transforms = get_transforms(cfg, 'train')
    val_transforms = get_transforms(cfg, 'val')
    
    print('Fold:', cfg['val_fold'])
    print('Number of training images by class:', Counter([data_list_train[i]['case_class'] for i in range(len(data_list_train))]))
    print('Number of validation images by class:', Counter([data_list_val[i]['case_class'] for i in range(len(data_list_val))]))
    
    train_dss = CacheDataset(np.array(data_list_train), transform=train_transforms, cache_rate=cfg['training']['cache_rate'], cache_num=sys.maxsize, num_workers=cfg['training']['num_workers'])
    val_dss = CacheDataset(np.array(data_list_val), transform=val_transforms, cache_rate = cfg['training']['cache_rate'], cache_num=sys.maxsize, num_workers=cfg['training']['num_workers'])
    
    train_loader = DataLoader(train_dss, batch_size=cfg['training']['train_batch_size'], shuffle=True, num_workers=cfg['training']['num_workers'], persistent_workers=True, pin_memory=torch.cuda.is_available()) 
    val_loader = DataLoader(val_dss, batch_size=cfg['training']['val_batch_size'], num_workers=cfg['training']['num_workers'], persistent_workers=True, pin_memory=torch.cuda.is_available())
    
    # check same train images
    results_fold_dir = os.path.join(cfg['results_dir'], f"{cfg['nfolds']}foldCV", cfg['model']['name'] + ('deep_supervision' if cfg['model']['name'] == 'DynUNet' and cfg['model']['params']['deep_supervision'] else '') + ('_'+ cfg['mode']['mode_type'] if len(cfg['mode']['mode_type']) > 1 else ''), cfg['preprocessing']['image_preprocess'],  f"{('fold_' + str(cfg['val_fold']) if cfg['val_fold'] != 'validation_cohort' else 'validation_cohort')}")
    os.makedirs(os.path.join(results_fold_dir, f'train_images_examples'), exist_ok=True)
    
    check_loader = train_loader
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["label"].shape)
    
    for i in range(len(check_data["img"])):
        check_image, check_label = (check_data["img"][i], check_data["label"][i])
        print(f"image shape: {check_image.shape}, label shape: {check_label.shape}")
        show_image(check_image, check_label, None, os.path.join(results_fold_dir, f'train_images_examples', f'train_sample_{i}.png'))   
        
    
    train(cfg, train_loader, val_loader, results_fold_dir)
        
    if cfg['wandb']['state']: wandb.finish()
        
if __name__ == "__main__":
    # cfg = "/home/benito/script/NephroBIT/KPIs24/config_train_swinUNETR.yaml"
    #define parser to pass the configuration file
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file", default="/home/benito/script/NephroBIT/KPIs24/configs/config_train_swinUNETR_noCV.yaml")
    args = parser.parse_args()
    cfg = args.config
    
    main(cfg)

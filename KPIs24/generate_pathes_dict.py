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
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.losses import DiceCELoss
from monai.apps.nuclick.transforms import SplitLabelMined
from monai.apps import CrossValidation
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.utils.data import Subset
from monai.transforms import (
    SaveImage
)
from losses import HoVerNetLoss
from collections import Counter
import matplotlib.pyplot as plt
from monai.utils import set_determinism
import wandb 
import random
from utilites import seed_everything, prepare_data, load_config, save_image_jpg, save_mask_jpg, show_image, get_transforms, get_model
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
torch.autograd.set_detect_anomaly(True)


def main(cfg):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    cfg = load_config(cfg)
    
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
    
    #get validation list files
    if cfg['val_fold'] is not None:
        datadir = os.path.join(cfg['datadir'], f"fold_{cfg['val_fold']}")
    else:
        datadir = cfg['datadir']
    data_list = prepare_data(datadir)
    
    print('Number of images:', len(data_list))
    
    train_transforms = get_transforms(cfg, 'generate_patches')
    
    if cfg['val_fold'] is not None: print('Fold:', cfg['val_fold'])
    print('Number of images by class:', Counter([data_list[i]['case_class'] for i in range(len(data_list))]))

    dss= CacheDataset(np.array(data_list), transform=train_transforms, cache_rate=0.5, cache_num=sys.maxsize, num_workers=4)
    
    loader = DataLoader(dss, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=torch.cuda.is_available()) 
    
    # check same train images
    
    if cfg['val_fold'] is not None: 
        results_directory = os.path.join(cfg['results_dir'], f"fold_{cfg['val_fold']}")
    else:
        results_directory = cfg['results_dir']
        
    os.makedirs(results_directory, exist_ok=True)
    
    for idx_img, batch_data in enumerate(loader):
        inputs, labels = batch_data["img"], batch_data["label"]
        #save inputs and labels as jpg
        for idx in range(inputs.shape[0]):
            #make dir
            os.makedirs(os.path.join(results_directory, batch_data['case_class'][idx],batch_data['case_id'][idx], 'img'), exist_ok=True)
            os.makedirs(os.path.join(results_directory, batch_data['case_class'][idx],batch_data['case_id'][idx], 'mask'), exist_ok=True)
            save_image_jpg(inputs[idx].permute(1, 2, 0).cpu().detach().numpy(), os.path.join(results_directory, batch_data['case_class'][idx],batch_data['case_id'][idx], 'img', f"{batch_data['img_path'][idx].split('/')[-1].split('.jpg')[0]}_{idx}.jpg"), mode = 'RGB')
            save_image_jpg((labels[idx].cpu().detach().numpy()[0]*255), os.path.join(results_directory, batch_data['case_class'][idx],batch_data['case_id'][idx], 'mask', f"{batch_data['label_path'][idx].split('/')[-1].split('.jpg')[0]}_{idx}.png"), mode = 'L')
            
        
if __name__ == "__main__":
    #define parser to pass the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file", default="/home/benito/script/NephroBIT/KPIs24/configs/config_generate_patches.yaml")
    args = parser.parse_args()
    cfg = args.config
    
    main(cfg)

import os
import sys
import yaml
import logging
import argparse

import numpy as np

from torch.cuda import is_available
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism

from collections import Counter

from utilities import load_config, prepare_data, seed_everything, get_transforms
from evaluation import evaluate_func


def main(cfg):

    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    cfg = load_config(cfg)
    
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
    
    #get validation list files
    datadir_val = cfg['datadir']
    data_list_val = prepare_data(datadir_val)
    
    if not cfg['ensemble']:
        val_transforms = get_transforms(cfg, 'test')
    else:
        #this is necessary because the function for ensemble need the key to be 'image' 
        for item in data_list_val:
            item['image'] = item.pop('img')
        
        val_transforms = get_transforms(cfg, 'ensemble')
    
    print('Number of test images by class:', Counter([data_list_val[i]['case_class'] for i in range(len(data_list_val))]))
    val_dss = CacheDataset(np.array(data_list_val), transform=val_transforms, cache_rate = cfg['validation']['cache_rate'], cache_num=sys.maxsize, num_workers=cfg['validation']['num_workers'])
    val_loader = DataLoader(val_dss, batch_size=cfg['validation']['val_batch_size'], num_workers=cfg['validation']['num_workers'], persistent_workers=True, pin_memory=is_available())
    
    results_dir = cfg['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    evaluate_func(cfg, val_loader, results_dir, save_masks=cfg['save_masks'])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file", 
                        default="/nephrobit/src/config.yaml")
    args = parser.parse_args()
    cfg = args.config
    main(cfg)

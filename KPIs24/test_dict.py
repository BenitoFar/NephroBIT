
import os, logging, sys, yaml
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import monai
import argparse
from monai.metrics import DiceMetric
from monai.engines import EnsembleEvaluator
from monai.data import decollate_batch, CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.apps.nuclick.transforms import SplitLabeld
from monai.handlers import MeanDice, from_engine
from monai.inferers import  SlidingWindowInferer
from monai.utils import set_determinism
import wandb 
from collections import Counter
from utilites import seed_everything, prepare_data, load_config, save_mask, get_model, get_transforms
from evaluate import evaluate_func

def main(cfg):
    monai.config.print_config()
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
    val_loader = DataLoader(val_dss, batch_size=cfg['validation']['val_batch_size'], num_workers=cfg['validation']['num_workers'], persistent_workers=True, pin_memory=torch.cuda.is_available())
    
    
    if cfg['ensemble']:
        results_dir = os.path.join(cfg['results_dir'], f"{cfg['nfolds']}foldCV", "ensemble_" +"_".join([l for l in cfg['model']['name']]), cfg['preprocessing']['image_preprocess'], 
                                    ('fold_' + str(cfg['val_fold']) if cfg['val_fold'] != 'validation_cohort' else 'validation_cohort'), 
                                    cfg['inference_type'], 
                                    cfg['validation']['sliding_window_inference']['mode'] + '_windowing',
                                    ("TTA" if cfg['validation']['timetestaugmentation']['status'] else "noTTA"),
                                    ('post_processing' if cfg['postprocessing']['status'] else 'no_post_processing'),
                                    ("_".join([l for l in cfg['postprocessing']['operations'] if cfg['postprocessing']['operations'][l]['status']]) if cfg['postprocessing']['status'] else ''),
                                    ('dice_include_background' if cfg['validation']['dice_include_background'] else 'dice_exclude_background'))
    else:
        results_dir = os.path.join(cfg['results_dir'], f"{cfg['nfolds']}foldCV", cfg['model']['name'], cfg['preprocessing']['image_preprocess'], 
                                    ("ensembleCV" if cfg['ensemble'] else f"{('fold_' + str(cfg['val_fold']) if cfg['val_fold'] != 'validation_cohort' else 'validation_cohort')}"), 
                                    cfg['inference_type'], 
                                    cfg['validation']['sliding_window_inference']['mode'] + '_windowing',
                                    ("TTA" if cfg['validation']['timetestaugmentation']['status'] else "noTTA"),
                                    ('post_processing' if cfg['postprocessing']['status'] else 'no_post_processing'),
                                    ("_".join([l for l in cfg['postprocessing']['operations'].keys() if cfg['postprocessing']['operations'][l]['status']]) if cfg['postprocessing']['status'] else ''), 
                                    ('dice_include_background' if cfg['validation']['dice_include_background'] else 'dice_exclude_background'))
    
    os.makedirs(results_dir, exist_ok=True)
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir, exist_ok=True)
    # else:
    #     results_dir = results_dir + '_v2'
    #     os.makedirs(results_dir, exist_ok=True)
    
    #save cfg in this folder
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as file:
        yaml.dump(cfg, file)
    
    # if cfg['wandb']['state']:
    #     run_name = f"{cfg['wandb']['group_name']}_{cfg['model']['name']}-fold{cfg['val_fold']:02}-inference-{cfg['inference_type']}_{cfg['validation']['sliding_window_inference']['mode']}_windowing"
    #     wandb.init(project=cfg['wandb']['project'], 
    #             name=run_name, 
    #             group= f"{cfg['wandb']['group_name']}_{cfg['model']['name']}_{cfg['nfolds']}foldcv_{cfg['preprocessing']['image_preprocess']}",
    #             entity = cfg['wandb']['entity'],
    #             save_code=True, 
    #             reinit=cfg['wandb']['reinit'], 
    #             resume=cfg['wandb']['resume'],
    #             config = cfg,
    #                 )
    
    evaluate_func(cfg, val_loader, results_dir, save_masks=cfg['save_masks'])
    
    # if cfg['wandb']['state']: wandb.finish()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file", default="/home/benito/script/NephroBIT/KPIs24/configs/config_test_DynUNet_noCV.yaml")
    args = parser.parse_args()
    cfg = args.config
    main(cfg)

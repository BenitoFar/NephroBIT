
import os, logging, sys
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import monai
import torch
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
from utilites import seed_everything, prepare_data, load_config, save_mask_jpg, get_model, get_transforms
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
    
    val_transforms = get_transforms(cfg, 'test')
    
    
    print('Number of test images by class:', Counter([data_list_val[i]['case_class'] for i in range(len(data_list_val))]))
    val_dss = CacheDataset(np.array(data_list_val), transform=val_transforms, cache_rate = cfg['validation']['cache_rate'], cache_num=sys.maxsize, num_workers=cfg['validation']['num_workers'])
    val_loader = DataLoader(val_dss, batch_size=cfg['validation']['val_batch_size'], num_workers=cfg['validation']['num_workers'], persistent_workers=True, pin_memory=torch.cuda.is_available())
    
    
    results_dir = os.path.join(cfg['results_dir'], f"{cfg['nfolds']}foldCV", cfg['model']['name'], cfg['preprocessing']['image_preprocess'], 
                               ("ensemble" if cfg['ensemble'] else f"fold_{cfg['val_fold']}"), 
                               cfg['inference_type'], 
                               cfg['validation']['sliding_window_inference']['mode'] + '_windowing',
                               ("TTA" if cfg['validation']['timetestaugmentation']['status'] else "noTTA"))
    
    os.makedirs(results_dir, exist_ok=True)
    
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
    
    evaluate_func(cfg, val_loader, results_dir, save_masks=False)
    
    # if cfg['wandb']['state']: wandb.finish()
    
if __name__ == "__main__":
    cfg = "/home/benito/script/NephroBIT/KPIs24/config_test_swinUNETR.yaml"
    main(cfg)

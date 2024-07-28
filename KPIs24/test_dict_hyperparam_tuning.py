
import os, logging, sys, yaml, time
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
import itertools

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
    
    # Define the sets of values you want to try for each parameter
    param_grid = {
        'min_size' : [200, 500, 1000, 2000],
        'kernel_size_oc' : [20, 50, 100, 200],
        'kernel_size_fill_holes' : [20, 50, 100, 200]
    }

    # Generate all combinations of parameter values
    param_combinations = list(itertools.product(*param_grid.values()))

    #create dataframe to save the results
    results_df = pd.DataFrame(index=range(len(param_combinations)), columns=['small_object_min_size', 'kernel_size_opening-closing', 'kernel_size_fill_holes', 'mean_dice', 'mean_hausdorff_distance', 'mean_dice_by_class', 'mean_hausdorff_distance_by_class'])
    # Iterate over all combinations of parameter values
    #calculate time of each combination
    for j, param_values in enumerate(param_combinations):
        time_start = time.time()
        # if param_values[0] == 100 and param_values[1] == 10:
        #     continue
        # Create a copy of the original cfg dictionary
        cfg_copy = cfg.copy()

        # Update the parameters in the cfg copy
        cfg_copy['postprocessing']['operations']['remove_small_components']['min_size'] = param_values[0]
        cfg_copy['postprocessing']['operations']['closing']['kernel_size'] = param_values[1]
        cfg_copy['postprocessing']['operations']['opening']['kernel_size'] = param_values[1]
        cfg_copy['postprocessing']['operations']['fill_holes']['kernel_size'] = param_values[2]
        
        #redifine the results_dir to save the results of each combination of parameters
        results_dir_actual = os.path.join(results_dir, 'hyperopt' ,f"remove_small_components_{cfg_copy['postprocessing']['operations']['remove_small_components']['min_size']}_closing_{cfg_copy['postprocessing']['operations']['closing']['kernel_size']}_opening_{cfg_copy['postprocessing']['operations']['opening']['kernel_size']}_fill_holes_{cfg_copy['postprocessing']['operations']['fill_holes']['kernel_size']}")
        os.makedirs(results_dir_actual, exist_ok=True)

        #save cfg in this folder
        with open(os.path.join(results_dir, 'config.yaml'), 'w') as file:
            yaml.dump(cfg_copy, file)
        
        df = evaluate_func(cfg_copy, val_loader, results_dir_actual, save_masks=cfg['save_masks'])
        results_df.loc[j,'small_object_min_size'] = param_values[0]
        results_df.loc[j, 'kernel_size_opening-closing'] = param_values[1]
        results_df.loc[j, 'kernel_size_fill_holes'] = param_values[2]
        results_df.loc[j, 'mean_dice'] = df['dice'].mean()
        results_df.loc[j, 'mean_hausdorff_distance'] = df['hausdorff_distance'].mean()
        results_df.loc[j, 'mean_dice_by_class'] = list(df.groupby('class')['dice'].mean().values)

        results_df.to_csv(os.path.join(results_dir, 'results_postprocessing_hyperparameters_selection.csv'), index=False)
        time_end = time.time()
        print(f'Elapsed time: {time_end - time_start} seconds')
    mean_dice_by_class = results_df.groupby('class')['dice'].mean()
    mean_dice = results_df.mean()
    class_count = df.groupby('class')['dice'].count()
    
    print('/n')
    print(f'Parameters combination {param_values[0]} - {param_values[1]}')
    print(f' Mean DICE: {mean_dice}')
    print("Mean dice by class: ", mean_dice_by_class, '- N = ', len(class_count))
    print("Count by class: ", class_count)
    print('/n')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file", default="/home/benito/script/NephroBIT/KPIs24/configs/config_test_DynUNet_noCV.yaml")
    args = parser.parse_args()
    cfg = args.config
    main(cfg)

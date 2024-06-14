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
import yaml
from collections import Counter
import nibabel as nib
import numpy as np
import torch
from ignite.metrics import Accuracy
from monai.metrics import DiceMetric
import monai
from monai.engines import EnsembleEvaluator, SupervisedEvaluator, SupervisedTrainer
from monai.data import decollate_batch, DataLoader, CacheDataset
from monai.inferers import sliding_window_inference
from monai.apps.nuclick.transforms import SplitLabeld
from monai.transforms import (
    Activationsd,
    Activations,
    AsDiscrete,
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    MeanEnsembled,
    KeepLargestConnectedComponentd,
    LoadImaged,
    SaveImaged,
    SaveImage,
    ScaleIntensityd,
    EnsureTyped,
    ScaleIntensityRangeD,
    VoteEnsembled,
)
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.metrics import compute_hausdorff_distance
from monai.visualize import plot_2d_or_3d_image
from monai.utils import set_determinism
import wandb 
from utilites import seed_everything, prepare_data, load_config, save_jpg_mask, get_model, get_transforms

def ensemble_evaluate(cfg, post_transforms, test_loader, models):
    evaluator = EnsembleEvaluator(
        device= torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu"),
        val_data_loader=test_loader,
        pred_keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
        networks=models,
        inferer=SlidingWindowInferer(roi_size=cfg['preprocessing']['roi_size'], sw_batch_size=1, overlap=0.5),
        postprocessing=post_transforms,
        key_val_metric={
            "test_mean_dice": MeanDice(
                include_background=True,
                output_transform=from_engine(["pred", "label"]),
            )
        },
    )
    evaluator.run()
    return evaluator.state.metrics

def evaluate(cfg, val_loader, results_dir):
    device = torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
    
    if cfg['ensemble']:
        #load all models
        models = []
        for fold in range(cfg['n_folds']):
            #if exists model path load model
            if os.path.exists(os.path.join(cfg['model_dir'], f"fold_{fold}","best_metric_model.pth")):
                model = get_model(cfg, os.path.join(cfg['model_dir'], "best_metric_fold{}.pth".format(fold)))
            else:
                #raise warning that model for fold does not exist
                print("Model for fold {} does not exist".format(fold))
            models.append(model)
    else:
        model = get_model(cfg, os.path.join(cfg['model_dir'], f"fold_{cfg['fold']}", "best_metric_model.pth"))
    
    if cfg['ensemble']:
        #evaluate each model 
        #create results folder for ensemble images
        results_dir = os.path.join(results_dir, "predicted_masks_ensemble")
        os.makedirs(results_dir, exist_ok=True)
        
        val_post_transforms = Compose(
            [   EnsureTyped(keys=["pred0", "pred1", "pred2", "pred3", "pred4"]),
                MeanEnsembled(
                    keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
                    output_key="pred",
                    # in this particular example, we use validation metrics as weights
                    weights=[1, 1, 1, 1, 1],
                ),
                Activationsd(keys= 'pred', sigmoid=True), 
                AsDiscreted(keys= 'pred', threshold=0.5),
                # KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
                # SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=os.path.join(cfg['model_dir'], "predicted_masks_ensemble")),
                ]
            )
    
        scores = ensemble_evaluate(cfg, val_post_transforms, val_loader, models)
            
        print("Metrics of ensemble (MEAN): {}".format(scores))
        
    else:
        results_dir = os.path.join(results_dir, "predicted_masks")
        os.makedirs(os.path.join(cfg['model_dir'], f"predictions_fold{cfg['fold']}"), exist_ok=True)
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        
        #define post transforms
        val_post_transforms = Compose(
            [
                Activations(sigmoid=True), 
                AsDiscrete(threshold=0.5),
                # KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
                SaveImage(meta_keys="image_meta_dict", output_dir=os.path.join(cfg['model_dir'], "predictions_fold0")),
                ]
            )
        
    with torch.no_grad():
        val_images = None
        val_labels = None
        val_outputs = None
        
        for idx_val, val_data in enumerate(val_loader):
            val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
            sw_batch_size = 1
            val_outputs = sliding_window_inference(val_images, cfg['preprocessing']['roi_size'], sw_batch_size, model)
            val_outputs = [val_post_transforms(i) for i in decollate_batch(val_outputs)]          
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            # dice_metric_batch(y_pred=val_outputs, y=val_labels)
            #save prediction mask as jpg
            save_jpg_mask(val_outputs, os.path.join(results_dir, f'{val_data["label_path"][0].split("/")[-1].split(".jpg")[0]}.jpg'))
            
        # aggregate the final mean dice result
        scores = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    print("Metric of fold {}: {}".format(fold, scores)) 
        
    if cfg['wandb']['state']: 
        wandb.log({"best_dice_validation_metric": dice_metric})
        wandb.finish()
           

def main(cfg):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    cfg = load_config(cfg)
    
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
    
    datadir = cfg['datadir']
    
    #get validation list files
    datadir_val = os.path.join(datadir)
        
    data_list_val = prepare_data(datadir_val)
    
    val_transforms = get_transforms(cfg, 'val')
    
    print('Fold:', cfg['val_fold'])
    print('Number of validation images by class:', Counter([data_list_val[i]['case_class'] for i in range(len(data_list_val))]))
    val_dss = CacheDataset(np.array(data_list_val), transform=val_transforms, cache_rate = cfg['training']['cache_rate'], cache_num=sys.maxsize, num_workers=cfg['training']['num_workers'])
    val_loader = DataLoader(val_dss, batch_size=cfg['training']['val_batch_size'], num_workers=cfg['training']['num_workers'], persistent_workers=True, pin_memory=torch.cuda.is_available())
    
    if cfg['ensemble']:
        results_dir = os.path.join(cfg['results_dir'], f"{cfg['nfolds']}foldCV", cfg['model']['name'], cfg['preprocessing']['image_preprocess'], "ensemble")
    else:
        results_dir = os.path.join(cfg['results_dir'], f"{cfg['nfolds']}foldCV", cfg['model']['name'], cfg['preprocessing']['image_preprocess'],  f"fold_{cfg['val_fold']}")
    
    if cfg['wandb']['state']:
        run_name = f"{cfg['wandb']['group_name']}_{cfg['model']['name']}-fold{cfg['val_fold']:02}-inference-{cfg['inference_type']}"
        wandb.init(project=cfg['wandb']['project'], 
                name=run_name, 
                group= f"{cfg['wandb']['group_name']}_{cfg['model']['name']}_{cfg['nfolds']}foldcv_{cfg['preprocessing']['image_preprocess']}",
                entity = cfg['wandb']['entity'],
                save_code=True, 
                reinit=cfg['wandb']['reinit'], 
                resume=cfg['wandb']['resume'],
                config = cfg,
                    )
    
    evaluate(cfg, val_loader, results_dir)
    
    if cfg['wandb']['state']: wandb.finish()
    
if __name__ == "__main__":
    cfg = "config_evaluation.yaml"
    main(cfg)

import os
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.engines import EnsembleEvaluator
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.apps.nuclick.transforms import SplitLabeld
from monai.handlers import MeanDice, from_engine
from monai.inferers import  SlidingWindowInferer
from monai.utils import set_determinism
import wandb 
from torchvision.transforms.functional import rotate
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from utilites import seed_everything, prepare_data, load_config, save_jpg_mask, get_model, get_transforms
import pandas as pd
from monai.transforms.utils import allow_missing_keys_mode
from monai.transforms.utils_pytorch_numpy_unification import mode, stack
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete, SaveImage
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch, TestTimeAugmentation
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureTyped, 
    MeanEnsembled,
    Activationsd,
    AsDiscreted,
    RandFlipd, 
    RandRotate90d
)

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

def evaluate_func(cfg, val_loader, results_dir, save_masks=False):
    device = torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
    progress = open(results_dir + '/progress_eval.txt', 'w')
    
    if cfg['ensemble']:
        #load all models
        models = []
        for fold in range(cfg['n_folds']):
            #if exists model path load model
            if os.path.exists(os.path.join(cfg['modeldir'], f"fold_{fold}","best_metric_model.pth")):
                model = get_model(cfg, os.path.join(cfg['modeldir'], "best_metric_fold{}.pth".format(fold)))
            else:
                #raise warning that model for fold does not exist
                print("Model for fold {} does not exist".format(fold))
            models.append(model)
    else:
        model = get_model(cfg, os.path.join(cfg['modeldir'], f"fold_{cfg['val_fold']}", "best_metric_model.pth"))
    
    if cfg['ensemble']:
        #evaluate each model 
        #create results folder for ensemble images
        if save_masks: 
            results_dir_masks = os.path.join(results_dir, "predicted_masks_ensemble")
            os.makedirs(results_dir_masks, exist_ok=True)
        val_post_transforms = Compose(
            [   EnsureTyped(keys=["pred0", "pred1", "pred2", "pred3", "pred4"]),
                MeanEnsembled(
                    keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
                    output_key="pred",
                    weights=[1, 1, 1, 1, 1],
                ),
                Activationsd(keys= 'pred', sigmoid=True), 
                AsDiscreted(keys= 'pred', threshold=0.5),
                ]
            )
    
        scores = ensemble_evaluate(cfg, val_post_transforms, val_loader, models)
            
        print("Metrics of ensemble (MEAN): {}".format(scores))
        
    else:
        if save_masks:
            results_dir_masks = os.path.join(results_dir, f"predicted_masks_fold{cfg['val_fold']}")
            os.makedirs(results_dir_masks, exist_ok=True)
       
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        
        #define post transforms
        val_post_transforms = Compose(
            [
                Activations(sigmoid=True), 
                AsDiscrete(threshold=0.5),
                ]
            )
    
    #save metrics to dataframe that has columns id and dice
    df = pd.DataFrame(columns=['id', 'class', 'id_patch', 'dice'])
    
    with torch.no_grad():
        val_images = None
        val_labels = None
        val_outputs = None
        
        for idx_val, val_data in enumerate(val_loader):
            val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
            sw_batch_size = 1
            if not cfg['validation']['timetestaugmentation']['status']:
                val_outputs = sliding_window_inference(val_images, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'])
                val_outputs = [val_post_transforms(i) for i in decollate_batch(val_outputs)]
            else:
                val_outputs = sliding_window_inference(val_images, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'])[0].detach().cpu()
                
                #apply model and then inverse the trasformations of each image
                val_images_aug1 = rotate(val_images, 90)
                val_images_aug2 = rotate(val_images, 180)
                val_images_aug3 = rotate(val_images, 270)
                val_images_aug4 = RandomHorizontalFlip(1)(val_images)
                val_images_aug5 = RandomVerticalFlip(1)(val_images)
                
                val_outputs_aug1 = sliding_window_inference(val_images_aug1, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'])[0].detach().cpu()
                val_outputs_aug2 = sliding_window_inference(val_images_aug2, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'])[0].detach().cpu()
                val_outputs_aug3 = sliding_window_inference(val_images_aug3, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'])[0].detach().cpu()
                val_outputs_aug4 = sliding_window_inference(val_images_aug4, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'])[0].detach().cpu()
                val_outputs_aug5 = sliding_window_inference(val_images_aug5, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'])[0].detach().cpu()
                
                #inverse the transformations
                val_outputs_aug1 = rotate(val_outputs_aug1, -90).to(device)
                val_outputs_aug2 = rotate(val_outputs_aug2, -180).to(device)
                val_outputs_aug3 = rotate(val_outputs_aug3, -270).to(device)
                val_outputs_aug4 = RandomHorizontalFlip(1)(val_outputs_aug4).to(device)
                val_outputs_aug5 = RandomVerticalFlip(1)(val_outputs_aug5).to(device)
                
                #take the mean of the 4 outputs
                val_outputs = stack([val_outputs, val_outputs_aug1, val_outputs_aug2, val_outputs_aug3, val_outputs_aug4, val_outputs_aug5], 0)
                val_outputs = val_outputs.mean(0) #mode(val_outputs, 0)
                val_outputs = [val_post_transforms(i) for i in decollate_batch(val_outputs)]
                #add dimension to val_outputs
                val_outputs = [torch.unsqueeze(i, 0) for i in val_outputs]
            # compute metric for current iteration
            actual_dice = dice_metric(y_pred=val_outputs, y=val_labels)
            #save prediction mask as jpg
            if save_masks: save_jpg_mask(val_outputs[0].cpu().numpy().squeeze(), os.path.join(results_dir_masks, f'{val_data["label_path"][0].split("/")[-1].split(".jpg")[0]}.jpg'))
            df.loc[idx_val, 'id'] = val_data["label_path"][0].split("/")[-4]
            df.loc[idx_val, 'id_patch'] = val_data["label_path"][0].split("/")[-1].split(".jpg")[0]
            df.loc[idx_val, 'class'] = val_data["label_path"][0].split("/")[-4]
            df.loc[idx_val, 'dice'] = actual_dice.cpu().numpy().squeeze()
            progress.write("Dice of image {}: {} \n".format(val_data["label_path"][0].split("/")[-1].split(".jpg")[0], actual_dice.cpu().numpy().squeeze()))
            
        # aggregate the final mean dice result
        scores = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    #save metrics to csv
    df.to_csv(os.path.join(results_dir, 'dice_metrics.csv'), index=False)
    
    print("Metric of fold {}: {}".format(cfg['val_fold'], scores)) 
    progress.write("Metric of fold {}: {} \n".format(cfg['val_fold'], scores))  
    
    mean_dice_by_class = df.groupby('class')['dice'].mean()
    # Print mean dice by class and count in progress
    class_count = df.groupby('class')['dice'].count()
    print("Mean dice by class: ", mean_dice_by_class, '- N = ', len(class_count))
    print("Count by class: ", class_count)
    
    progress.write("Mean dice by class: {}".format(mean_dice_by_class))
    progress.write("Count by class: {}".format(class_count))
    
    if cfg['wandb']['state']: 
        wandb.log({"best_dice_validation_metric": dice_metric})
        wandb.finish()
           


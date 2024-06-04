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

import nibabel as nib
import numpy as np
import torch
from ignite.metrics import Accuracy

import monai
from monai.apps import get_logger
from monai.data import create_test_image_3d
from monai.engines import SupervisedEvaluator, EnsembleEvaluator
from monai.handlers import CheckpointLoader, MeanDice, StatsHandler, from_engine
from monai.inferers import SlidingWindowInferer
from monai.apps.nuclick.transforms import SplitLabeld
from monai.transforms import (
    Activationsd,
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    MeanEnsembled,
    KeepLargestConnectedComponentd,
    LoadImaged,
    SaveImaged,
    ScaleIntensityd,
    EnsureTyped,
    ScaleIntensityRangeD,
    VoteEnsembled,
)
from monai.metrics import compute_hausdorff_distance
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
    
def prepare_data(datadir):
    """prepare data list"""
    images = sorted(glob(os.path.join(datadir, "**/*img.jpg"), recursive = True))[0:10]
    labels = sorted(glob(os.path.join(datadir, "**/*mask.jpg"), recursive = True))[0:10]
    
    data_list = [
        {"img": _image, "label": _label}
        for _image, _label in zip(images, labels)
    ]
    return data_list


def evaluate(cfg, val_loader, device, models, ensemble=False):
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
    
    roi_size = cfg['evaluate']['roi_size']
    sw_batch_size = cfg['evaluate']['sw_batch_size']


    model = monai.modelworks.models.Umodel(
            spatial_dims=cfg['model']['spatial_dims'],
            in_channels=cfg['model']['in_channels'],
            out_channels=cfg['model']['out_channels'],
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
    
    models = []
    for fold in range(4):
        model.load_state_dict(torch.load(
            os.path.join(cfg['model_dir'], "best_metric_model_fold{}.pth".format(fold))))
        model.eval()
        models.append(model)
        
    # post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    
    if ensemble:
        #evaluate each model 
        for fold in range(4):
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, models[i])
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                scores = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

            print("Metric of fold {}: {}".format(fold, scores))
    
        #define post transforms for ensemble
        val_post_transforms = Compose(
            [
                EnsureTyped(keys=["pred0", "pred1", "pred2", "pred3", "pred4"]),
                MeanEnsembled(
                    keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
                    output_key="pred",
                    # in this particular example, we use validation metrics as weights
                    weights=[0.95, 0.94, 0.95, 0.94, 0.90],
                ),
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold=0.5),
                KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
                SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir="./runs/"),
            ]
        )
        
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                all_val_outputs = []
                for i in range(4):
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, models[i])
                    all_val_outputs.append(val_outputs)

                all_val_outputs = torch.stack(all_val_outputs, dim=0)
                all_val_outputs = torch.mean(all_val_outputs, dim=0)
                
                post_pred_results = [val_post_transforms(i) for i in decollate_batch(all_val_outputs)]
                post_label_results = [post_label(i) for i in decollate_batch(val_labels)]
                
                # compute metric for current iteration
                dice_metric(y_pred=post_pred_results, y=post_label_results)

            # aggregate the final mean dice result
            weights = [0.95, 0.94, 0.95, 0.94, 0.90],
            
            # Aggregate the final mean dice result with weights
            scores = dice_metric.aggregate(weights).item()
            
            # reset the status for next validation round
            dice_metric.reset()

        print("Metric of ensemble (MEAN): {}".format(scores))

        
        # val_post_transforms = Compose(
        #     [
        #         EnsureTyped(keys=["pred0", "pred1", "pred2", "pred3", "pred4"]),
        #         Activationsd(keys=["pred0", "pred1", "pred2", "pred3", "pred4"], sigmoid=True),
        #         # transform data into discrete before voting
        #         AsDiscreted(keys=["pred0", "pred1", "pred2", "pred3", "pred4"], threshold=0.5),
        #         VoteEnsembled(keys=["pred0", "pred1", "pred2", "pred3", "pred4"], output_key="pred"),
        #         KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
        #         SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir="./runs/"),
        #     ]
        # )
    
    else:
        #define post transforms
        val_post_transforms = Compose(
            [
                EnsureTyped(keys="pred"),
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold=0.5),
                KeepLargestConnectedComponentd(keys="pred", applied_labels=[1]),
                SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir="./runs/"),
            ]
        )
        
        model.load_state_dict(torch.load(
                os.path.join(cfg['model_dir'], "best_metric_model_fold{}.pth".format(fold))))
        model.eval()
        
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            scores = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

        print("Metric of fold {}: {}".format(fold, scores)) 
        
        wandb.log({"best_dice_validation_metric": dice_metric})
        
    

def load_config(cfg):
    with open(cfg, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(cfg):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # get_logger("eval_log")
    config = load_config(cfg)
    
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
     
    device = torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu")
    
    datadir = config['datadir']
    results_dir = config['results_dir']
    ensemble = config['ensemble']
    
    data_list = prepare_data(datadir)
    
    val_transforms = Compose(
            [
                LoadImaged(keys=["img", "label"]),
                EnsureChannelFirstd(keys=["img"], channel_dim=-1),
                EnsureChannelFirstd(keys=["label"], channel_dim='no_channel'),
                SplitLabeld(keys="label", mask_value=None, others_value=255),
                ScaleIntensityRangeD(keys="img", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ]
    )
    
    # create a validation data loader
    val_ds = monai.data.Dataset(data=data_list, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4)

    #load models from the model directory
    models_paths = glob(os.path.join(results_dir, "*.pth"))
    
    #load models
    models = []
    for model_path in models_paths:
        model = monai.modelworks.models.Umodel(
            spatial_dims=cfg['model']['spatial_dims'],
            in_channels=cfg['model']['in_channels'],
            out_channels=cfg['model']['out_channels'],
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
            
        model = model.load_state_dict(torch.load(model_path))
        models.append(model)
    
    wandb.init(project=cfg['wandb']['project'], 
                   name=f'validation_{('ensemble' if ensemble else 'best_cv_model')}', 
                   group=f'cv_{cfg['wandb']['group_name']}',
                   entity = cfg['wandb']['entity'],
                   save_code=True, 
                   reinit=cfg['wandb']['reinit'], 
                   resume=cfg['wandb']['resume'],
                   config = cfg,
                    )
    
    evaluate(cfg, val_loader, device, models, ensemble=ensemble)
    wandb.finish()
    
if __name__ == "__main__":
    cfg = "config_evaluation.yaml"
    main(cfg)

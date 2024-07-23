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
from utilites import seed_everything, prepare_data, load_config, save_mask, get_model, get_transforms
import pandas as pd
from monai.transforms.utils import allow_missing_keys_mode
from monai.transforms.utils_pytorch_numpy_unification import mode, stack
from monai.metrics import DiceMetric, GeneralizedDiceScore, HausdorffDistanceMetric
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
)
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from skimage.morphology import binary_opening, binary_closing
from skimage.measure import label, regionprops
from monai.utils import convert_to_numpy
from monai.transforms import MapTransform
from monai.config import KeysCollection
from post_processing import crf

def ensemble_evaluate(cfg, post_transforms, test_loader, models):
    evaluator = EnsembleEvaluator(
        device= torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu"),
        val_data_loader=test_loader,
        pred_keys=[f"pred{i}" for i in range(len(cfg['models_list']))],
        networks=models,
        inferer=SlidingWindowInferer(roi_size=cfg['preprocessing']['roi_size'], sw_batch_size=1, mode= cfg['validation']['sliding_window_inference']['mode'], progress = True),
        postprocessing=post_transforms,
        key_val_metric={"test_mean_dice": MeanDice(
                                                    include_background=cfg['validation']['dice_include_background'],
                                                    output_transform=from_engine(["pred", "label"]),
                                                    reduction="mean", 
                                                    save_details = True)
            }
    )
    evaluator.run()
    return evaluator


class PostProcessLabels(MapTransform):
    """
    Remove small connected components from the prediction using binary morphology.

    Args:
        min_size (int): minimum size to keep the component.
        connectivity (int): the connectivity for the connected component.
        keys (KeysCollection): Keys of the data dictionary to apply the transform on.
    """
    
    def __init__(
        self,
        operations: list, 
        keys: KeysCollection,
    ):
        super().__init__(keys, allow_missing_keys=False)
        self.operations = operations
    def __call__(self, data):
        d = dict(data)

        if len(self.keys) > 1:
            print("Only 'pred' key is supported, more than 1 key was found")
            return None

        for key in self.keys:
            
            pred = d[key] if isinstance(d[key], torch.Tensor) else torch.from_numpy(d[key])

            pred_array = pred.cpu().numpy().squeeze()

            if self.operations['remove_small_components']['status']:
                # Calculate connected components
                connected_components = label(pred_array)

                #get labels of the components touching the borders 
                border_components_labels = set(connected_components[0, :].tolist() + connected_components[-1, :].tolist() + connected_components[:, 0].tolist() + connected_components[:, -1].tolist())
                
                # Calculate region properties of the connected components
                props = regionprops(connected_components)

                # Remove components with less than 100 pixels
                for prop in props:
                    if prop.area < self.operations['remove_small_components']['min_size']:
                        if prop.label not in border_components_labels or prop.area < 20:
                            connected_components[connected_components == prop.label] = 0
                pred_array = connected_components
                pred_array[pred_array > 0] = 1

            # You can use binary morphology operations or any other method of your choice here
            if self.operations['closing']['status']:
                pred_array = binary_closing(pred_array, footprint =np.ones((self.operations['closing']['kernel_size'], self.operations['closing']['kernel_size'])))
                           
            if self.operations['opening']['status']:
                pred_array = binary_opening(pred_array, footprint =np.ones((self.operations['opening']['kernel_size'], self.operations['opening']['kernel_size'])))
            
            if self.operations['crf']['status']:
                pred_array = crf(d['image'], pred_array)
                
            # Convert back to torch.Tensor
            pred = torch.from_numpy(pred_array).to(pred.device)
            pred = pred.unsqueeze(0).unsqueeze(0)

            # You can use binary morphology operations or any other method of your choice here
            d[key] = pred if isinstance(d[key], torch.Tensor) else convert_to_numpy(pred)
                        
        return d
    
def evaluate_func(cfg, val_loader, results_dir, save_masks=False):
    device = torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
    progress = open(results_dir + '/progress_eval.txt', 'w')
    
    #save metrics to dataframe that has columns id and dice
    df = pd.DataFrame(columns=['id', 'class', 'id_patch', 'dice', 'hausdorff_distance'])
    
    if save_masks: 
        results_dir_masks = os.path.join(results_dir, "predicted_masks")
        os.makedirs(results_dir_masks, exist_ok=True)
            
    if cfg['ensemble']:
        #load all models
        models = []
        models = get_model(cfg, cfg['models_list'])
        
        #evaluate each model 
    
        if cfg['postprocessing']['status']:
            val_post_transforms = Compose(
                [   EnsureTyped(keys=[f"pred{i}" for i in range(len(cfg['models_list']))]),
                    MeanEnsembled(
                        keys=[f"pred{i}" for i in range(len(cfg['models_list']))],
                        output_key="pred",
                        weights=[1 for i in range(len(cfg['models_list']))],
                    ),
                    Activationsd(keys= 'pred', sigmoid=True), 
                    AsDiscreted(keys= 'pred', threshold=0.5),
                    PostProcessLabels(keys='pred', operations=cfg['postprocessing']['operations']), #cfg['postprocessing']['min_size']
                    ]
                )
        else:
            val_post_transforms = Compose(
                [   EnsureTyped(keys=[f"pred{i}" for i in range(len(cfg['models_list']))]),
                    MeanEnsembled(
                        keys=[f"pred{i}" for i in range(len(cfg['models_list']))],
                        output_key="pred",
                        weights=[1 for i in range(len(cfg['models_list']))],
                    ),
                    Activationsd(keys= 'pred', sigmoid=True), 
                    AsDiscreted(keys= 'pred', threshold=0.5),
                    ]
                )

        scores = ensemble_evaluate(cfg, val_post_transforms, val_loader, models)
            
        print("Metrics of ensemble (MEAN): {}".format(scores.state.metrics))
        
        scores_array = scores.state.metric_details['test_mean_dice'].cpu().numpy().squeeze()
    
        df['id'] = [i['case_id'] for i in scores.data_loader.dataset.data]
        df['class'] = [i['case_class'] for i in scores.data_loader.dataset.data]   
        df['id_patch'] = [i['img_path'].split("/")[-1].split(".jpg")[0] for i in scores.data_loader.dataset.data]
        df['dice'] = scores_array
        scores = scores.state.metrics['test_mean_dice']
    else:
        model = get_model(cfg, os.path.join(cfg['modeldir'], f"{('fold_' + str(cfg['val_fold']) if cfg['val_fold'] != 'validation_cohort' else 'validation_cohort')}", "best_metric_model.pth"))
        
        dice_metric = DiceMetric(include_background=cfg['validation']['dice_include_background'], reduction="mean", get_not_nans=False)
        
        hausdorff_distance_metric = HausdorffDistanceMetric(include_background=cfg['validation']['dice_include_background'], reduction="mean")
        
        #define post transforms
        if cfg['postprocessing']['status']:
            val_post_transforms = Compose(
                [  
                    Activationsd(keys='pred',sigmoid=True), 
                    AsDiscreted(keys='pred',threshold=0.5),
                    PostProcessLabels(keys=['pred'], operations=cfg['postprocessing']['operations']), #cfg['postprocessing']['min_size']
                    ]
                )
        else:
            val_post_transforms = Compose(
                [  
                    Activationsd(keys='pred',sigmoid=True), 
                    AsDiscreted(keys='pred',threshold=0.5),
                    ]
                )
    
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            
            for idx_val, val_data in enumerate(val_loader):
                val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                sw_batch_size = 1
                val_outputs_orig = sliding_window_inference(val_images, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'], device = device)
                
                if cfg['validation']['timetestaugmentation']['status']:
                    #apply model and then inverse the trasformations of each image
                    val_images_aug1 = rotate(val_images, 90)
                    val_images_aug2 = rotate(val_images, 180)
                    val_images_aug3 = rotate(val_images, 270)
                    val_images_aug4 = RandomHorizontalFlip(1)(val_images)
                    val_images_aug5 = RandomVerticalFlip(1)(val_images)
                    
                    val_outputs_aug1 = sliding_window_inference(val_images_aug1, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'], device = device)
                    val_outputs_aug2 = sliding_window_inference(val_images_aug2, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'], device = device)
                    val_outputs_aug3 = sliding_window_inference(val_images_aug3, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'], device = device)
                    val_outputs_aug4 = sliding_window_inference(val_images_aug4, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'], device = device)
                    val_outputs_aug5 = sliding_window_inference(val_images_aug5, cfg['preprocessing']['roi_size'], sw_batch_size, model, mode = cfg['validation']['sliding_window_inference']['mode'], device = device)
                    
                    #inverse the transformations
                    val_outputs_aug1 = rotate(val_outputs_aug1, -90)
                    val_outputs_aug2 = rotate(val_outputs_aug2, -180)
                    val_outputs_aug3 = rotate(val_outputs_aug3, -270)
                    val_outputs_aug4 = RandomHorizontalFlip(1)(val_outputs_aug4)
                    val_outputs_aug5 = RandomVerticalFlip(1)(val_outputs_aug5)
                    
                    #take the mean of the 4 outputs
                    val_outputs = stack([val_outputs_orig, val_outputs_aug1, val_outputs_aug2, val_outputs_aug3, val_outputs_aug4, val_outputs_aug5], 0)
                    val_outputs = val_outputs.mean(0, keepdim = True) #mode(val_outputs, 0) or sum(val_outputs)/len(val_outputs)
                else:
                    val_outputs = val_outputs_orig
                
                #save prediction probabilies numpy array as npy file
                if cfg['save_probabilities']: 
                    os.makedirs(os.path.join(results_dir, "predicted_probabilities"), exist_ok=True)
                    np.save(os.path.join(results_dir, "predicted_probabilities", f'{val_data["label_path"][0].split("/")[-1].split(".jpg")[0]}.npy'), val_outputs.cpu().numpy().squeeze())
                
                #apply post transforms
                val_outputs = [val_post_transforms({'pred': val_outputs , 'image': val_images})['pred'] for i in decollate_batch(val_outputs)]
                                
                # compute metric for current iteration
                actual_dice = dice_metric(y_pred=val_outputs, y=val_labels)

                actual_hausdorff_distance = hausdorff_distance_metric(y_pred=val_outputs[0], y=val_labels)
                
                #save prediction mask as jpg
                if save_masks: save_mask((val_outputs[0].cpu().numpy().squeeze()*255).astype('uint8'), os.path.join(results_dir_masks, f'{val_data["label_path"][0].split("/")[-1].split(".jpg")[0]}.png'))
                df.loc[idx_val, 'id'] = val_data["label_path"][0].split("/")[-3]
                df.loc[idx_val, 'id_patch'] = val_data["label_path"][0].split("/")[-1].split(".jpg")[0]
                df.loc[idx_val, 'class'] = val_data["label_path"][0].split("/")[-4]
                df.loc[idx_val, 'dice'] = actual_dice.cpu().numpy().squeeze()
                
                df.loc[idx_val, 'hausdorff_distance'] = actual_hausdorff_distance.cpu().numpy().squeeze()
                progress.write("Dice of image {}: {} \n".format(val_data["label_path"][0].split("/")[-1].split(".jpg")[0], actual_dice.cpu().numpy().squeeze()))
                
            # aggregate the final mean dice result
            scores = dice_metric.aggregate().item()
            
            scores_hausdorff = hausdorff_distance_metric.aggregate().item()
            
            # reset the status for next validation round
            dice_metric.reset()

    #save metrics to csv
    df.to_csv(os.path.join(results_dir, 'dice_metrics.csv'), index=False)
    
    print("Metric of fold {}: {}".format(cfg['val_fold'], scores)) 
    progress.write("Metric of fold {}: {} \n".format(cfg['val_fold'], scores))  
    
    print("Hausdorff Distance of fold {}: {}".format(cfg['val_fold'], scores_hausdorff))
    progress.write("Hausdorff Distance of fold {}: {} \n".format(cfg['val_fold'], scores_hausdorff))
    
    mean_dice_by_class = df.groupby('class')['dice'].mean()
    # Print mean dice by class and count in progress
    class_count = df.groupby('class')['dice'].count()
    print("Mean dice by class: ", mean_dice_by_class, '- N = ', len(class_count))
    print("Count by class: ", class_count)
    
    progress.write("Mean dice by class: {}".format(mean_dice_by_class))
    
    mean_hausdorff_distance_by_class = df.groupby('class')['hausdorff_distance'].mean()
    class_count = df.groupby('class')['hausdorff_distance'].count()
    print("Mean hausdorff distance by class: ", mean_hausdorff_distance_by_class, '- N = ', len(class_count))
    progress.write("Mean hausdorff distance by class: {}".format(mean_hausdorff_distance_by_class))
                   
    progress.write("Count by class: {}".format(class_count))
    
    if cfg['wandb']['state']: 
        wandb.log({"best_dice_validation_metric": dice_metric})
        wandb.finish()
    
    return df     


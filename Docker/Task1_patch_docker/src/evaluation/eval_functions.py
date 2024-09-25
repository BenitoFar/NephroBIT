import os
import sys

import numpy as np
import pandas as pd

import torch

from monai.engines import EnsembleEvaluator
from monai.handlers import MeanDice, from_engine
from monai.inferers import  SlidingWindowInferer, sliding_window_inference
from monai.utils import set_determinism, convert_to_numpy

from torchvision.transforms.functional import rotate
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

from monai.transforms.utils_pytorch_numpy_unification import stack
from monai.metrics import DiceMetric
from monai.transforms import Compose
from monai.data import decollate_batch, FolderLayoutBase, MetaTensor
from monai.transforms import (
    Compose,
    EnsureTyped, 
    MeanEnsembled,
    Activationsd,
    AsDiscreted,
    MapTransform,
    SaveImaged,
    ScaleIntensityd
)
from monai.config import KeysCollection, PathLike

from skimage.morphology import binary_opening, binary_closing
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utilities import seed_everything, save_mask, get_model


class CustomFolderLayout(FolderLayoutBase):

    def __init__(
        self,
        output_dir: PathLike,
        postfix: str = "",
        extension: str = "",
        parent: bool = False,
        makedirs: bool = False,
        data_root_dir: PathLike = "",
    ):
        """
        Args:
            output_dir: output directory.
            postfix: a postfix string for output file name appended to ``subject``.
            extension: output file extension to be appended to the end of an output filename.
            parent: whether to add a level of parent folder to contain each image to the output filename.
            makedirs: whether to create the output parent directories if they do not exist.
            data_root_dir: an optional `PathLike` object to preserve the folder structure of the input `subject`.
                Please see :py:func:`monai.data.utils.create_file_basename` for more details.
        """
        self.output_dir = output_dir
        self.postfix = postfix
        self.ext = extension
        self.parent = parent
        self.makedirs = makedirs
        self.data_root_dir = data_root_dir

    def filename(self, subject: PathLike, **kwargs) -> PathLike:

        
        # get the filename and directory
        filedir, filename = os.path.split(subject)
        # remove extension
        filename, ext = os.path.splitext(filename)
        if ext == ".gz":
            filename, ext = os.path.splitext(filename)
        # use data_root_dir to find relative path to file
        filedir_rel_path = ""
        if self.data_root_dir and filedir:
            filedir_rel_path = os.path.relpath(filedir, self.data_root_dir)

        # output folder path will be original name without the extension
        output = os.path.join(self.output_dir, filedir_rel_path)

        if self.parent:
            output = os.path.join(output, filename)

        output = os.path.dirname(output) # remove /img/ folder in path

        if self.makedirs:
            # create target folder if no existing
            os.makedirs(output, exist_ok=True)

        # remove last subfolder (/img/) in filename
        filename = '_'.join(filename.split('_')[:-1])

        # add the sub-folder plus the postfix name to become the file basename in the output path
        if self.postfix != "":
            filename += f'_{self.postfix}'

        output = os.path.normpath(os.path.join(output, filename))

        
        if self.ext is not None:
            ext = f"{self.ext}"
            output += f".{ext}" if ext and not ext.startswith(".") else f"{ext}"

        return output


def ensemble_evaluate(cfg, post_transforms, test_loader, models):
    evaluator = EnsembleEvaluator(
        device= torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu"),
        val_data_loader=test_loader,
        pred_keys=[f"pred{i}" for i in range(len(cfg['models_list']))],
        networks=models,
        inferer=SlidingWindowInferer(roi_size=cfg['preprocessing']['roi_size'], sw_batch_size=1, mode= cfg['validation']['sliding_window_inference']['mode'], progress = True),
        postprocessing=post_transforms,
        key_val_metric={
            "test_mean_dice": MeanDice(
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

        for key in self.key_iterator(d):

            if isinstance(d[key], MetaTensor):
                pred_array = d[key].get_array()
            
            else:
                pred = d[key] if isinstance(d[key], torch.Tensor) else torch.from_numpy(d[key])
                pred_array = pred.cpu().numpy()
            
            pred_array = np.squeeze(pred_array)

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
            if self.operations['fill_holes']['status']:
                pred_array = binary_fill_holes(pred_array, structure = np.ones((self.operations['fill_holes']['kernel_size'], self.operations['fill_holes']['kernel_size'])))
             
             
            if self.operations['closing']['status']:
                pred_array = binary_closing(pred_array, footprint =np.ones((self.operations['closing']['kernel_size'], self.operations['closing']['kernel_size'])))
                           
            if self.operations['opening']['status']:
                pred_array = binary_opening(pred_array, footprint =np.ones((self.operations['opening']['kernel_size'], self.operations['opening']['kernel_size'])))
               
            pred_array = np.expand_dims(pred_array, axis=0)

            # Convert back to MetaTensor / Tensor
            if isinstance(d[key], MetaTensor):
                d[key].set_array(pred_array)
            else:
                pred = torch.from_numpy(pred_array).to(pred.device)
                d[key] = pred if isinstance(d[key], torch.Tensor) else convert_to_numpy(pred)
        
        return d
    
    
def evaluate_func(cfg, val_loader, results_dir, save_masks=False):
    device = torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])

    if cfg['compute_metrics']:
        # progress = open(results_dir + '/progress_eval.txt', 'w')
        #save metrics to dataframe that has columns id and dice
        df = pd.DataFrame(columns=['id', 'class', 'id_patch', 'dice'])
            
    if cfg['ensemble']:
        #load all models
        models = []
        models = get_model(cfg, cfg['models_list'])
        
        #evaluate each model 
    
        if cfg['postprocessing']['status']:
            val_post_transforms = Compose(
                [   
                    EnsureTyped(keys=[f"pred{i}" for i in range(len(cfg['models_list']))]),
                    MeanEnsembled(
                        keys=[f"pred{i}" for i in range(len(cfg['models_list']))],
                        output_key="pred",
                        weights=[1 for i in range(len(cfg['models_list']))],
                    ),
                    Activationsd(keys= 'pred', sigmoid=True), 
                    AsDiscreted(keys= 'pred', threshold=0.5),
                    PostProcessLabels(keys='pred', operations=cfg['postprocessing']['operations']), #cfg['postprocessing']['min_size']
                    ScaleIntensityd(keys="pred", maxv=255, dtype=np.uint8),
                    SaveImaged(
                        keys="pred",
                        folder_layout=CustomFolderLayout(
                            output_dir=results_dir,
                            postfix="mask",
                            extension="png",
                            parent=False,
                            makedirs=True,
                            data_root_dir=cfg['datadir'],
                        ),
                        output_ext=".png",
                        output_dtype="uint8",
                    ),
                    
                ]
            )
        else:
            val_post_transforms = Compose(
                [   
                    EnsureTyped(keys=[f"pred{i}" for i in range(len(cfg['models_list']))]),
                    MeanEnsembled(
                        keys=[f"pred{i}" for i in range(len(cfg['models_list']))],
                        output_key="pred",
                        weights=[1 for i in range(len(cfg['models_list']))],
                    ),
                    Activationsd(keys= 'pred', sigmoid=True), 
                    AsDiscreted(keys= 'pred', threshold=0.5),
                    ScaleIntensityd(keys="pred", maxv=255, dtype=np.uint8),
                    SaveImaged(
                        keys="pred",
                        folder_layout=CustomFolderLayout(
                            output_dir=results_dir,
                            postfix="mask",
                            extension="png",
                            parent=False,
                            makedirs=True,
                            data_root_dir=cfg['datadir'],
                        ),
                        output_ext=".png",
                        output_dtype="uint8",
                    ),
                ]
            )
        
        scores = ensemble_evaluate(cfg, val_post_transforms, val_loader, models)

        if cfg['compute_metrics']:
            print("Metrics of ensemble (MEAN): {}".format(scores.state.metrics))
            scores_array = scores.state.metric_details['test_mean_dice'].cpu().numpy().squeeze()
            df['id'] = [i['case_id'] for i in scores.data_loader.dataset.data]
            df['class'] = [i['case_class'] for i in scores.data_loader.dataset.data]   
            df['id_patch'] = [i['img_path'].split("/")[-1].split(".jpg")[0] for i in scores.data_loader.dataset.data]
            df['dice'] = scores_array
            scores = scores.state.metrics['test_mean_dice']

    else:
        model = get_model(cfg, os.path.join(cfg['modeldir'], f"{('fold_' + str(cfg['val_fold']) if cfg['val_fold'] != 'validation_cohort' else 'validation_cohort')}", "best_metric_model.pth"))
        
        if cfg['compute_metrics']:
            dice_metric = DiceMetric(include_background=cfg['validation']['dice_include_background'], reduction="mean", get_not_nans=False)
        
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
                    results_probability_dir = os.path.join(results_dir, "probabilities" , f'{val_data["case_class"][0]}', f'{val_data["case_id"][0]}', 'mask')
                    os.makedirs(results_probability_dir, exist_ok=True)
                    np.save(os.path.join(results_probability_dir, f'{val_data["label_path"][0].split("/")[-1].split(".jpg")[0]}.npy'), val_outputs.cpu().numpy().squeeze())
                
                #apply post transforms
                val_outputs = [val_post_transforms({'pred': val_outputs , 'image': val_images})['pred'] for i in decollate_batch(val_outputs)]
                                
                # compute metric for current iteration
                if cfg['compute_metrics']:
                    actual_dice = dice_metric(y_pred=val_outputs, y=val_labels)
                
                #save prediction mask as jpg
                if save_masks: 
                    results_logits_dir = os.path.join(results_dir , f'{val_data["case_class"][0]}', f'{val_data["case_id"][0]}')
                    os.makedirs(results_logits_dir, exist_ok=True)
                    save_mask((val_outputs[0].cpu().numpy().squeeze()*255).astype('uint8'), os.path.join(results_logits_dir, f'{val_data["label_path"][0].split("/")[-1].split(".jpg")[0]}.png'))
                
                if cfg['compute_metrics']:
                    df.loc[idx_val, 'id'] = val_data["case_id"][0]
                    df.loc[idx_val, 'id_patch'] = val_data["label_path"][0].split("/")[-1].split(".jpg")[0]
                    df.loc[idx_val, 'class'] = val_data["case_class"][0]
                    df.loc[idx_val, 'dice'] = actual_dice.cpu().numpy().squeeze()
                    
                    # progress.write("Dice of image {}: {} \n".format(val_data["label_path"][0].split("/")[-1].split(".jpg")[0], actual_dice.cpu().numpy().squeeze()))

            if cfg['compute_metrics']: 
                # aggregate the final mean dice result
                scores = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

    if cfg['compute_metrics']:
        #save metrics to csv
        df.to_csv(os.path.join(results_dir, 'dice_metrics.csv'), index=False)
        
        print("Metric of fold {}: {}".format(cfg['val_fold'], scores)) 
        # progress.write("Metric of fold {}: {} \n".format(cfg['val_fold'], scores))  
        
        mean_dice_by_class = df.groupby('class')['dice'].mean()
        # Print mean dice by class and count in progress
        class_count = df.groupby('class')['dice'].count()
        print("Mean dice by class: ", mean_dice_by_class, '- N = ', len(class_count))
        print("Count by class: ", class_count)
        
        # progress.write("Mean dice by class: {}".format(mean_dice_by_class))  
        # progress.write("Count by class: {}".format(class_count))
    
        return df

    return

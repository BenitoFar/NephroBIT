
"""Import modules required to run the Jupyter notebook."""
#model source -> \\138.4.55.135\data_KPIs\data\Results_segmentation\Task1_patch_level\3foldCV\SwinUNETR\CropPosNeg\fold_0
from __future__ import annotations

# Clear logger to use tiatoolbox.logger
import logging
import warnings

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import torch
from matplotlib import cm
from pathlib import Path
import PIL
from PIL import Image
import shutil
from argparse import ArgumentParser

from tiatoolbox import logger
from tiatoolbox.models.architecture.unet import UNetModel
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
)
from tiatoolbox.utils.misc import download_data, imread
from tiatoolbox.utils.visualization import overlay_prediction_mask
from tiatoolbox.wsicore.wsireader import WSIReader

# Torch-related
from torchvision import transforms
from torcheval.metrics.functional import binary_f1_score as f1_score

#Monai
from monai.networks.nets import SwinUNETR, DynUnet
from monai.inferers import  SimpleInferer
from monai.transforms import Compose, Activations, AsDiscrete, SaveImage
from monai.data import decollate_batch
import types

#WSI processing
import openslide
from sklearn.metrics import f1_score
import pandas as pd
import os
#import subprocess
import gc
#postprocess
import tifffile as tifi
from tiatoolbox.utils.transforms import imresize
from tiatoolbox.tools.patchextraction import PatchExtractor
import cv2

import json 
ON_GPU = True  # Should be changed to False if no cuda-enabled GPU is available

def get_folders_with_wsi_tiff(path):
    folder_list = []
    # Iterate through the items in the directory
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            # Check if any file in the directory ends with "_wsi.tiff"
            contains_wsi_tiff = any(fname.endswith("_wsi.tiff") for fname in os.listdir(item_path))
            if contains_wsi_tiff:
                folder_list.append(item)
    return folder_list

def load_model(config, preproc_func, infer_batch):
    model_class = globals()[config['class']]
    model_params = config['params']
    checkpoint_path = config['checkpoint_path']
    
    model = model_class(**model_params)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.preproc_func = preproc_func
    model.infer_batch = infer_batch
    
    return model

def preproc_func(img):
    img = PIL.Image.fromarray(img)
    img = transforms.ToTensor()(img)
    return img

def infer_batch(
        model,#: nn.Module,
        batch_data,#: torch.Tensor,
        *,
        on_gpu: bool,
        ensemble: bool,
    ) -> list[np.ndarray]:
        img_patches_device = batch_data.to('cuda') if on_gpu else batch_data
        if ensemble:
            for m in model:                 
                m.eval()

            with torch.inference_mode():
                inferer = SimpleInferer()
                
                # Perform inference with the first model to initialize the output
                output = inferer(inputs=img_patches_device, network=model[0])
                
                # Sum outputs from all models
                for m in model[1:]:
                    output += inferer(inputs=img_patches_device, network=m)
                
                # Average the outputs
                output /= len(model)                
                output = torch.permute(output, (0, 2, 3, 1))
            
        else:
            model.eval()
            with torch.inference_mode():
                inferer = SimpleInferer()
                output = inferer(inputs=img_patches_device, network=model)
                output = torch.permute(output, (0,2,3,1))
                
        return [output.cpu().numpy()] 

def postproc_func(output) -> list[np.ndarray]:
    
    post_transforms = Compose(
            [
                Activations(sigmoid=True), 
                AsDiscrete(threshold=0.5),                 
                ]
            )
    output = torch.from_numpy(output).to('cuda')
    output_post = post_transforms(output)  
    return output_post.cpu().detach().numpy()[:,:,0] 

def process_mask(mask, kernel_size, min_region_size):
    """
    Process the mask to remove small regions and apply morphological dilation.

    Parameters:
    mask (numpy.ndarray): Input binary mask.
    kernel_size (tuple, list, int): Size of the kernel for morphological operations.
    min_region_size (int): Minimum size of regions to keep.

    Returns:
    numpy.ndarray: Processed mask.
    """
    # Ensure the mask is binary
    mask = (mask > 0).astype(np.uint8)

    # Find connected components with statistics
    num_labels, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Get the sizes of the components
    sizes = stats[1:, -1]

    # Remove small regions
    for i, size in enumerate(sizes):
        if size < min_region_size:
            if np.any(output == i + 1 & (mask == 0)):
                mask[output == i + 1] = 0

    # Ensure kernel size is valid
    kernel_size_tuple = tuple(np.round(np.repeat(np.array(kernel_size), 2)).astype(int))
     
    # Create structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size_tuple)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def process_big_image(image, kernel_size, min_region_size, patch_size, overlap,postp):
    """
    Process a big image by splitting it into smaller patches with overlapping,
    applying the process_mask function to each patch, and merging the results.

    Parameters:
    image (numpy.ndarray): Input image.
    kernel_size (tuple, list, int): Size of the kernel for morphological operations.
    min_region_size (int): Minimum size of regions to keep.
    patch_size (tuple, list, int): Size of each patch.
    overlap (float): Overlap percentage between patches.

    Returns:
    numpy.ndarray: Processed image.
    """
    
    (patch_coords, _) = PatchExtractor.get_coordinates(
                image_shape=image[:,:,0].shape,
                patch_input_shape=patch_size,
                patch_output_shape=patch_size,
                stride_shape=patch_size,
            )
    # Create an empty array to store the results
    result = np.zeros_like(image[:,:,0])

    # Iterate over the patches
    for patch_c in range(patch_coords.shape[0]):                 
        # Calculate the patch boundaries
        patch_x_start = max(0, patch_coords[patch_c][0])
        patch_x_end = min(image.shape[1], patch_coords[patch_c][2])
        patch_y_start = max(0,  patch_coords[patch_c][1])
        patch_y_end = min(image.shape[0], patch_coords[patch_c][3])

        # Extract the patch
        patch = image[patch_y_start:patch_y_end, patch_x_start:patch_x_end,:]
        patch = postproc_func(patch)
        if postp:
            # Apply the process_mask function to the patch
            processed_patch = process_mask(patch, kernel_size, min_region_size)
        else:
            processed_patch = patch       

        # Apply the processed patch to the result
        result[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = processed_patch
            
    return result

def main(cfg):     
    subtypes = get_folders_with_wsi_tiff(cfg["input_path"]) #['normal', 'NEP25', 'DN', '56Nx']    
    warnings.filterwarnings("ignore")    
    
    # Check if experiment folder already exists
    if os.path.exists(cfg['exp_name']):
         if cfg['remove_exp']:
             shutil.rmtree(cfg['exp_name'])
              
         else:
             print("ERROR: Experiment folder already exists!")
             exit()    
    os.makedirs(cfg['exp_name']) 
    
    if not os.path.exists(cfg['output_mask']):   
        os.makedirs(cfg['output_mask'])
            
    # Check if wsi_path is a folder
    if os.path.isdir(cfg['input_path']):         
        cfg['input_path'] = [Path(os.path.join(cfg['input_path'], fold, file)) for fold in subtypes for file in os.listdir(os.path.join(cfg['input_path'], fold)) if file.endswith('_wsi.tiff') ]                       
         
    else:
        cfg['input_path'] = Path(cfg['input_path'])
        
    for subtype in subtypes:
        os.makedirs(os.path.join(cfg['output_mask'], subtype), exist_ok=True)
        
    # Load model
    with open(cfg['config_path'], 'r') as f:
        model_configs = json.load(f)
        
    if cfg['model_name'] == 'all':
        model = []
        for model_name in model_configs.keys():
            model.append(load_model(model_configs[model_name], preproc_func, infer_batch))
    elif cfg['model_name'] in model_configs:
        model = load_model(model_configs[cfg['model_name']], preproc_func, infer_batch)
    else:
        print("ERROR: Model not found")
        exit()       
    
    bcc_segmentor = SemanticSegmentor(
        model=model,
        batch_size=cfg['bs'],
        model_list = cfg['model_name'] == 'all',
    )
    
    print(f"Processing {len(cfg['input_path'])} files")
    wsi_output = []
    exp_names = [cfg['exp_name']+"/"+str(i) for i in range(len(cfg['input_path']))]
    for exp in range(len(exp_names)):        
        wsi_output_l = bcc_segmentor.predict(
            [cfg['input_path'][exp]],  
            save_dir=Path(exp_names[exp]), 
            mode="tile",
            resolution=1.0,
            units="baseline",
            patch_input_shape=[512, 512],
            patch_output_shape=[512, 512],
            stride_shape=[384,384],#0.25 overlap
            on_gpu=ON_GPU,
            crash_on_exception=True,
        )        
        wsi_output.extend(wsi_output_l)
    
    subtype = "" 
    f1_scores = []
    case_names=[]
    subtype_dic = {name: [] for name in subtypes} #{"56Nx": [], "DN": [], "NEP25": [], "normal": []}
    for case in wsi_output:            
        # [img_origen ->0, img_predicha -> 1]
        input_path = Path(case[0])
        folder_name = input_path.parent.name
        if folder_name != subtype:
            subtype = folder_name
            print("#############################")
            print("Processing subtype: ", subtype)                           
        case_name = input_path.stem.replace('_wsi', '')
                       
        wsi_prediction_raw = np.load(case[1] + ".raw.0.npy", mmap_mode='r+' )  
        wsi_prediction = process_big_image(wsi_prediction_raw, kernel_size = 10, min_region_size=100, patch_size = (10000, 10000), overlap = 0, postp = cfg['postp'])
        if subtype != "NEP25":
            wsi_prediction = imresize(wsi_prediction, scale_factor=1/2, output_size=(wsi_prediction.shape[1], wsi_prediction.shape[0]))
        wsi_prediction_to_save = wsi_prediction *255
        wsi_prediction_to_save_np = wsi_prediction_to_save.astype(np.uint8)
        tifi.imwrite(cfg['output_mask']+'/'+subtype+ '/' + case_name + "_mask.tiff", wsi_prediction_to_save_np, dtype=np.uint8)

        if cfg['compute_metrics']:
            print("Computing metrics")            
            gt_file_path =str(input_path.parent) + "/"+case_name + "_mask.tiff"     
            gt_read = openslide.OpenSlide(gt_file_path)        
            gt = gt_read.read_region((0,0), 0, (gt_read.level_dimensions[0]))
            
            wsi_file_path =input_path   
            wsi_read = openslide.OpenSlide(wsi_file_path)        
            wsi = wsi_read.read_region((0,0), 0, (wsi_read.level_dimensions[0]))
            if not os.path.isdir(cfg['output_mask']+'/'+subtype+'/thumbnails'):  
                os.makedirs(cfg['output_mask']+'/'+subtype+'/thumbnails')
                        
            # Save thumbnails
            wsi_r = wsi.resize((int(wsi.size[0]*0.10), int(wsi.size[1]*0.10)))
            pred_r = Image.fromarray(wsi_prediction_to_save_np).convert('RGB').resize((int(wsi.shape[0]*0.10), int(wsi.shape[1]*0.10)))
            gt_r = gt.convert('RGB').resize((int(wsi.size[0]*0.10), int(wsi.size[1]*0.10))) 
            plt.subplot(1, 3, 1).set_title("image"), plt.imshow(wsi_r )
            plt.xticks([])  # Removes x-axis numbers and ticks
            plt.yticks([])  # Removes y-axis numbers and ticks
            plt.subplot(1, 3, 2).set_title("label"), plt.imshow(gt_r)
            plt.xticks([])  # Removes x-axis numbers and ticks
            plt.yticks([])  # Removes y-axis numbers and ticks
            plt.subplot(1, 3, 3).set_title("prediction"), plt.imshow(pred_r)    
            plt.xticks([])  # Removes x-axis numbers and ticks
            plt.yticks([])  # Removes y-axis numbers and ticks
            plt.savefig(cfg['output_mask']+'/'+subtype+'/thumbnails/'+case_name+'.png', bbox_inches='tight', dpi=300)
            
            y_true = (gt/255).astype(np.uint8)
            y_pred = wsi_prediction.astype(np.uint8)
            f1 = f1_score(y_true.flatten(), y_pred.flatten())#f1_score(torch.from_numpy(y_true.flatten()).to("cuda"), torch.from_numpy(y_pred.flatten()).to("cuda"))
            f1_scores.append(f1)
            subtype_dic[subtype].append(f1)
            case_names.append(case_name)
            print(f'Case {case_name}: F1 Score = {f1:.3f}')
        
    if cfg['compute_metrics']:
        print(f'Mean F1 Score = {np.mean(f1_scores):.3f}')
        for subtype in subtype_dic:
            print(f'Subtype {subtype}: Mean F1 Score = {np.mean(subtype_dic[subtype]):.3f}')
        df = pd.DataFrame({
            'Case': [case_names],
            'F1 Score': f1_scores
        })

        # Save to a CSV file
        df.to_csv(cfg['output_mask']+'/f1_scores.csv', index=False)

    if cfg['remove_dir']:
             shutil.rmtree(cfg['exp_name'],ignore_errors=True)
              
def parse_arguments():
    parser = ArgumentParser(description="KPI task 2: Inference for WSIs")

    parser.add_argument("--input_path", type=str, default="/input", help="input folder of WSIs") # /home/anadelia/atlas/kpi_t2/data/KPIs24_Validation_Data/Task2_WSI_level
    parser.add_argument("--output_mask", type=str, default="/output", help="Output folder of masks") # /home/anadelia/atlas/kpi_t2/data/Results_task2/output_ex_dynamic_unet2
    parser.add_argument("--config_path", type=str, default="/nephrobit/src/model_configs.json", help="Path to model config")       
    
    parser.add_argument("--exp_name", type=str, default="nephrobit_tmpresults",help="Folder to save results")
    parser.add_argument("--model_name", type=str, default="all", help="Model name: dynamic_unet, swin_unetr, all(ensemble)")  
    parser.add_argument("--bs",  type=int, default=16, help="batch size, originally 16")
    parser.add_argument("--compute_metrics", type=bool, default=False, help="Compute F1-score")    
    parser.add_argument("--postp", type=bool, default=False, help="If post-processing is required") #CHECK
    parser.add_argument("--remove_exp", type=bool, default=True, help="If experiment folder exists, remove it")
    parser.add_argument("--remove_dir", type=bool, default=True, help="Remove experiment directory after inference")
    
    args = parser.parse_args()
    config_dict = vars(args)

    return config_dict


if __name__ == "__main__":
    cfg = parse_arguments()
    main(cfg)
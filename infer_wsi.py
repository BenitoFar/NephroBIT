
"""Import modules required to run the Jupyter notebook."""
#model source -> \\138.4.55.135\data_KPIs\data\Results_segmentation\Task1_patch_level\3foldCV\SwinUNETR\CropPosNeg\fold_0
from __future__ import annotations

# Clear logger to use tiatoolbox.logger
import logging
import warnings

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import matplotlib as mpl
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
import subprocess

ON_GPU = True  # Should be changed to False if no cuda-enabled GPU is available

def rmdir(dir_path: str | Path) -> None:
    """Helper function to delete directory."""
    if Path(dir_path).is_dir():
        shutil.rmtree(dir_path)
        logger.info("Removing directory %s", dir_path)


def preproc_func(img):
    img = PIL.Image.fromarray(img)
    img = transforms.ToTensor()(img)
    return img

def infer_batch(
        model,#: nn.Module,
        batch_data,#: torch.Tensor,
        *,
        on_gpu: bool,
    ) -> list[np.ndarray]:
        img_patches_device = batch_data.to('cuda') if on_gpu else batch_data
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

def main(cfg):
    subtypes = ["56Nx", "DN", "NEP25", "normal"] #igual puedo poner condición de "all" o especificar subtipo
 
    warnings.filterwarnings("ignore")    
    # Check if experiment folder already exists
    if os.path.exists(cfg['exp_name']):
         if cfg['remove_exp']:
             #shutil.rmtree(cfg['exp_name'])
             subprocess.run(['rm', '-r', cfg['exp_name']], check=True, text=True, capture_output=True)
         else:
             print("ERROR: Experiment folder already exists!")
             #exit()     
    if not os.path.exists(cfg['output_mask']):   
        os.makedirs(cfg['output_mask'])
            
    # Check if wsi_path is a folder
    if os.path.isdir(cfg['input_path']):         
        cfg['input_path'] = [Path(os.path.join(cfg['input_path'], fold, file)) for fold in subtypes for file in os.listdir(os.path.join(cfg['input_path'], fold)) if file.endswith('_wsi.tiff') ]               
        #cfg['input_path'] = cfg['input_path'][0] #CHECK
    else:
        cfg['input_path'] = Path(cfg['input_path'])
    for subtype in subtypes:
        os.makedirs(os.path.join(cfg['output_mask'], subtype), exist_ok=True)
    # Load model
    if cfg['model_name'] == 'swin_unetr':
        model = SwinUNETR(img_size=(512,512), in_channels=3, spatial_dims= 2, out_channels=1, feature_size=24, num_heads= [3, 6, 12, 24], depths = [2, 2, 2, 2], use_v2 = False)
        checkpoint = torch.load("/home/anadelia/atlas/kpi/data/Results_segmentation/Task1_patch_level/nofoldCV/SwinUNETR/CropPosNeg-ResizeLargerPatch/validation_cohort/best_metric_model.pth")  
        model.load_state_dict(checkpoint['model_state_dict'])
    elif cfg['model_name'] == 'dynamic_unet':
        model = DynUnet(img_size=(512,512), in_channels=3, spatial_dims= 2, out_channels=1, deep_supervision=False, res_block=True)
        checkpoint = torch.load("/home/anadelia/atlas/kpi/data/Results_segmentation/Task1_patch_level/nofoldCV/DynUNet/CropPosNeg-ResizeLargerPatch/validation_cohort/best_metric_model.pth")#(cfg['weights_path'])  
        model.load_state_dict(checkpoint['model_state_dict']) 
    else:
        print("ERROR: Model not found")
        #exit()     
    
    
    model.preproc_func = preproc_func
    model.infer_batch = infer_batch

    bcc_segmentor = SemanticSegmentor(
        model=model,
        batch_size=cfg['bs'],
    )
    print(f"Processing {len(cfg['input_path'])} files")
    wsi_output = bcc_segmentor.predict(
        cfg['input_path'],  #Path(cfg['input_path']),
        save_dir=Path(cfg['exp_name']), #global_save_dir,
        mode="tile",
        resolution=1.0,
        units="baseline",
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
        stride_shape=[384,384],#0.25 de overlap  #[512, 512],
        on_gpu=ON_GPU,
        crash_on_exception=True,
    )
    
    subtype = "" 
    f1_scores = []
    case_names=[]
    subtype_dic = {"56Nx": [], "DN": [], "NEP25": [], "normal": []}
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
        wsi_prediction = postproc_func(wsi_prediction_raw) 
        wsi_prediction_to_save = wsi_prediction *255
        wsi_prediction_to_save	= Image.fromarray(wsi_prediction_to_save.astype(np.uint8)) 
        if subtype != "NEP25":
            new_size = (wsi_prediction_to_save.width//2, wsi_prediction_to_save.height//2)
            wsi_prediction_to_save = wsi_prediction_to_save.resize(new_size)
        wsi_prediction_to_save.save(cfg['output_mask']+'/'+subtype+ '/' + case_name + "_mask.tiff")

        if cfg['compute_metrics']:
            print("Computing metrics")            
            gt_file_path =Path(input_path.parent + "/" + case_name + "_mask.tiff")     
            gt_read = openslide.OpenSlide(gt_file_path)        
            gt = gt_read.read_region((0,0), 0, (gt_read.level_dimensions[0])) #origin, level, dimensions    
            
            wsi_file_path =input_path   
            wsi_read = openslide.OpenSlide(wsi_file_path)        
            wsi = wsi_read.read_region((0,0), 0, (wsi_read.level_dimensions[0]))
            
            if not os.path.isdir(cfg['output_mask']+'/'+subtype+'/thumbnails'):  
                os.makedirs(cfg['output_mask']+'/'+subtype+'/thumbnails')
                        
            # Save thumbnails
            wsi_r = wsi.resize((int(imawsige.size[0]*0.10), int(wsi.size[1]*0.10)))
            pred_r = wsi_prediction_to_save.convert('RGB').resize((int(wsi.size[0]*0.10), int(wsi.size[1]*0.10)))
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
            
            y_true = (np.array(gt)[:,:,0]/255).astype(np.uint8)
            y_pred = wsi_prediction.astype(np.uint8)
            f1 = f1_score(torch.from_numpy(y_true.flatten()).to("cuda"), torch.from_numpy(y_pred.flatten()).to("cuda"))
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

        
def parse_arguments():
    parser = ArgumentParser(description="KPI task 2: Inference for WSIs")

    parser.add_argument("--input_path", type=str, default="/home/anadelia/atlas/kpi/data/KPIs24_Validation_Data/Task2_WSI_level", help="input folder of WSIs")
    parser.add_argument("--output_mask", type=str, default="/home/anadelia/atlas/kpi/data/Results_task2/output_ex_swinunetr", help="Output folder of masks")
    
    parser.add_argument("--exp_name", type=str, default="/home/anadelia/atlas/kpi/data/Results_task2/ex_swinunetr",help="Folder to save results")
    parser.add_argument("--model_name", type=str, default="swin_unetr", help="Model name: dynamic_unet, swin_unetr, unet, nnunet")
    #parser.add_argument("--weights_path", type=str, default="/home/anadelia/kpi_t2/swin_unetr_fold0.pth", help="Path to model weights")
    
    parser.add_argument("--bs",  type=int, default=4, help="batch size, originally 16")
    parser.add_argument("--compute_metrics", type=bool, default=True, help="Compute F1-score")    
    parser.add_argument("--postp", type=bool, default=False, help="If post-processing is required") #CHECK
    parser.add_argument("--remove_dir", type=bool, default=False, help="Remove experiment directory after inference")
    parser.add_argument("--remove_exp", type=bool, default=True, help="If experiment folder exists, remove it")
    #parser.add_argument("--path_size", type=float, default=512, help="Path size")
    #parser.add_argument("--overlap", type=float, default=0.25, help="Overlap for inference")
    
    args = parser.parse_args()
    config_dict = vars(args)

    return config_dict


if __name__ == "__main__":
    cfg = parse_arguments()

    main(cfg)
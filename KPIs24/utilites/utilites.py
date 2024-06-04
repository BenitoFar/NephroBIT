import numpy as np
import yaml
import os
from glob import glob
import random
import torch
from matplotlib import pyplot as plt


def show_image(image, label, predictions = None, filename = None):
    # print(f"Image: {image.shape}; Label: {label.shape}")

    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, 2)
        label = np.moveaxis(label, 0, 2)
        if predictions is not None:
            predictions = np.moveaxis(predictions, 0, 2)
            
    fig = plt.figure(figsize=(12, 6))
    if predictions is not None:
        n_subplots = 3
    else:
        n_subplots = 2    
    
    plt.subplot(1, n_subplots, 1)  
    plt.title("image")
    plt.imshow(image, cmap="gray")

    # if label is not None:
    #     masked = np.ma.masked_where(label == 0, label)
    #     plt.imshow(masked, "jet", interpolation="none", alpha=0.5)
    
    # plt.colorbar()

    if label is not None:
        plt.subplot(1, n_subplots, 2)
        plt.title("label")
        plt.imshow(label, cmap="gray", interpolation="none")
        # plt.colorbar()
    
    if predictions is not None:
        plt.subplot(1, n_subplots, 3)
        plt.title("prediction")
        plt.imshow(predictions, cmap="gray")
        # plt.colorbar()
    
    if filename:
        plt.savefig(filename)
    plt.close()
    
def load_config(cfg):
    with open(cfg, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_log_dir(cfg):
    log_dir = cfg["log_dir"] + '_' + cfg["model"]["name"]
    # if cfg["stage"] == 0:
    #     log_dir = os.path.join(log_dir, "stage0")
    print(f"Logs and models are saved at '{log_dir}'.")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def prepare_data(datadir):
    """prepare data list"""
    #get 100 random values between 0 and 2000
    images = sorted(glob(os.path.join(datadir, "**/*img.jpg"), recursive = True))
    labels = sorted(glob(os.path.join(datadir, "**/*mask.jpg"), recursive = True))
    
    data_list = [
        {"img": _image, "label": _label, 'img_class': os.path.dirname(_image).split('/')[-3], 'img_id': os.path.dirname(_image).split('/')[-2]}
        for _image, _label in zip(images, labels)
    ]
    return data_list

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    
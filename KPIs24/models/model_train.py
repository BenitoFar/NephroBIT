import os
import numpy as np
import random
import time
import torch
import monai
from monai.data import decollate_batch
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.optimizers import WarmupCosineSchedule
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from losses import HoVerNetLoss
from collections import Counter
import matplotlib.pyplot as plt
from monai.utils import set_determinism
import wandb 
import random
from utilites import seed_everything, save_mask, show_image, get_model
torch.autograd.set_detect_anomaly(True)

def train(cfg, train_loader, val_loader, results_dir):
    device = torch.device(f"cuda:{cfg['device_number']}" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=cfg['seed'])
    seed_everything(cfg['seed'])
    
    results_images_dir = os.path.join(results_dir, 'train_images')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_images_dir, exist_ok=True)
    progress = open(results_dir + '/progress_train.txt', 'w')
    
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    print(f"train_ds length: {len(train_ds)}, val_ds length: {len(val_ds)}")
    
    #define metrics
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # create UNet, DiceLoss and Adam optimizer
    model = get_model(cfg, pretrained_path = None)
    print('Number of parametets in the model:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        
    if cfg['wandb']['state']: wandb.watch(model, log="all")

    if cfg['training']['loss'] == 'DiceCELoss':
        loss_function =  DiceCELoss(sigmoid=True, lambda_dice=cfg['training']['loss_params']['lambda_dice'], lambda_ce=cfg['training']['loss_params']['lambda_ce'])
    elif cfg['training']['loss'] == 'HoVerNetLoss':
        loss_function = HoVerNetLoss(lambda_ce=0.8, lambda_dice=0.2, hovermaps=cfg['model']['params']['hovermaps'])
    else:
        raise ValueError(f"Loss {cfg['training']['loss']} not implemented")
    
    optimizer = torch.optim.Adam(model.parameters(), cfg['model']['optimizer']['params']['learning_rate'])
    
    if cfg['model']['scheduler']['name'] == "WarmupCosineSchedule":
        warmup_steps = int(cfg['model']['scheduler']['params']['warmup_epochs'] * len(train_ds) / cfg['training']['train_batch_size']) #train_ds.__ln__()
        t_total = int(cfg['training']['epoch'] * len(train_ds) / cfg['training']['train_batch_size'])
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total, cycles = cfg['model']['scheduler']['params']['cycles'], end_lr=1e-9)
    elif cfg['model']['scheduler']['name'] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg['model']['scheduler']['params']['T_max'], eta_min=1e-9)
    else:
        raise ValueError(f"Scheduler {cfg['model']['scheduler']['name']} not implemented")
    
    
    #get random indexes for validation data to show predicted images
    val_indexes = random.sample(range(len(val_ds)), 20)
    
    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    epoch_start_time = time.time()
    
    if cfg['resume_training']:
            checkpoint = torch.load(os.path.join(results_dir, "last_epoch_model.pth"))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            #save state rng of torch, np and random
            torch.set_rng_state(checkpoint['torch_rng_state'])
            np.random.set_state(checkpoint['np_rng_state'])
            random.setstate(checkpoint['random_rng_state'])
            first_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {first_epoch}")
    else:
        first_epoch = 0
        
    for epoch in range(first_epoch, cfg['training']['epoch']):
        epoch_start = time.time()
        
        print("-" * 10)
        print(f"epoch {epoch + 1}/{cfg['training']['epoch']}")
        model.train()
        epoch_loss = 0
        step = 0
        for idx_img, batch_data in enumerate(train_loader):
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device) 
            #save image and label
            if idx_img < 10:
                show_image(inputs[0].cpu().numpy(), labels[0].cpu().numpy(), None, os.path.join(results_images_dir, f'train_image_{idx_img}_{epoch}_{step}.png'))
            optimizer.zero_grad()
            outputs = model(inputs)
            # if idx_img < 10:
            #     show_image(inputs[0].cpu().numpy(), outputs[0].cpu().detach().numpy(), None, os.path.join(results_images_dir, f'train_outputs_{idx_img}_{epoch}_{step}.png'))
            if cfg['model']['name'] == 'DynUNet' and cfg['model']['params']['deep_supervision']:
                outputs = torch.unbind(outputs, dim=1)
                loss = sum([0.5**i * loss_function(output, labels) for i, output in enumerate(outputs)]) #can be passed also a weigth for each output feature map
            else:
                loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            # step scheduler after each epoch (cosine decay)
            scheduler.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // cfg['training']['train_batch_size']
            # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            # print("train_loss", loss.item(), " - ", epoch_len * epoch + step, "batches processed")
            
            # Update wandb dict
            if cfg['wandb']['state']:
                wandb.log({"train/loss": loss.item()})
                wandb.log({"learning_rate": scheduler.get_lr()[0]})
            progress.write(f'(epoch:{epoch+1} >> loss:{loss.item()}\n') 
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        
        print(f"epoch {epoch + 1} average train loss: {epoch_loss:.4f} - learning rate: {scheduler.get_lr()[0]:.5f} - time: {time.time() - epoch_start:.2f} seconds")

        #save model after each epoch 
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            #save rng state of torch, np and random
            'torch_rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate()
        }, os.path.join(results_dir, "last_epoch_model.pth"))
        
        if cfg['wandb']['state']:
            # 🐝 log train_loss averaged over epoch to wandb
            wandb.log({"train/loss_epoch": epoch_loss, "epoch": epoch+1})
            wandb.log({"epoch": epoch+1})
            # 🐝 log learning rate after each epoch to wandb
            wandb.log({"learning_rate": scheduler.get_lr()[0]})
             
        progress.write(f'(epoch:{epoch+1} >> train/loss_epoch:{epoch_loss}\n') 
        progress.write(f'(epoch:{epoch+1} >> learning_rate:{scheduler.get_lr()[0]}\n')
        progress.write(f'(epoch:{epoch+1} >> time:{time.time() - epoch_start:.2f} seconds\n')
        if (epoch + 1) % cfg['training']['val_interval'] == 0:
            epoch_start = time.time()
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                
                for idx_val, val_data in enumerate(val_loader):
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    sw_batch_size = 1
                    val_outputs = sliding_window_inference(val_images, cfg['preprocessing']['roi_size'], sw_batch_size, model)
                    #save output label image
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    if (idx_val in val_indexes) and (epoch) % 4 == 0:
                        show_image(val_images[0].cpu().numpy(), val_labels[0].cpu().numpy(), val_outputs[0].cpu().numpy(), os.path.join(results_images_dir,f'prediction_image_{idx_val}_epoch_{epoch+1}.png'))
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
            
                # Update wandb dict
                if cfg['wandb']['state']:
                    wandb.log({"val/dice_metric": metric})
                    #save
                progress.write(f'(epoch:{epoch+1} >> val/dice_metric:{metric}\n')
                   
                # reset the status for next validation round
                dice_metric.reset()
                
                metric_values.append(metric)
                
                if metric >= best_metric: #should it be >=?
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            #save rng state of torch, np and random
                            'torch_rng_state': torch.get_rng_state(),
                            'np_rng_state': np.random.get_state(),
                            'random_rng_state': random.getstate()                   
                        }, os.path.join(results_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                            f"\nbest mean dice: {best_metric:.4f}"
                            f" at epoch: {best_metric_epoch}"
                        )
                print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
                                     
    print(f"train completed, best_metric DICE: {best_metric:.4f} at epoch: {best_metric_epoch} - time consuming: {(time.time() - epoch_start_time):.4f}")
    if cfg['wandb']['state']:
        wandb.log({"best_dice_metric": best_metric, "best_metric_epoch": best_metric_epoch})
        wandb.save('model_best.pth')
    progress.write(f'(train completed, best_metric DICE: {best_metric:.4f} at epoch: {best_metric_epoch}\n')

    #load the best model and save all the validation prediction masks
    checkpoint = torch.load(os.path.join(results_dir, "best_metric_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    #create a folder to save the validation images
    os.makedirs(os.path.join(results_dir, 'predicted_masks'), exist_ok=True)
    
    with torch.no_grad():
        val_images = None
        val_labels = None
        val_outputs = None
        
        for idx_val, val_data in enumerate(val_loader):
            val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
            sw_batch_size = 1
            val_outputs = sliding_window_inference(val_images, cfg['preprocessing']['roi_size'], sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            #save output label image
            save_mask(val_outputs, os.path.join(results_dir, 'predicted_masks', f'{val_data["label_path"][0].split("/")[-1].split(".jpg")[0]}.jpg'))
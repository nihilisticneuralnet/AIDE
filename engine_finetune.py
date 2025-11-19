# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from utils import adjust_learning_rate
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score, 
    accuracy_score
)
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import matplotlib.cm
import cv2
import numpy as np
import os

# def save_cam_visualization(rgb_image, cams, file_name, output_dir, pred_prob, true_label, pred_label):
#     """
#     Save CAM visualizations as a grid without pytorch_grad_cam dependency
    
#     Args:
#         rgb_image: [3, H, W] tensor
#         cams: dict of {branch: heatmap [H, W]}
#         file_name: str
#         output_dir: str
#         pred_prob: float
#         true_label: int
#         pred_label: int
#     """
#     # Convert RGB image to numpy [H, W, 3] in range [0, 1]
#     rgb_img = rgb_image.permute(1, 2, 0).cpu().numpy()
#     rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
#     rgb_img = np.clip(rgb_img, 0, 1)
    
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     fig.suptitle(f'{file_name}\nPred: {pred_label} ({pred_prob:.3f}) | True: {true_label}', 
#                  fontsize=14)
    
#     # Original image
#     axes[0, 0].imshow(rgb_img)
#     axes[0, 0].set_title('Original')
#     axes[0, 0].axis('off')
    
#     # CAM overlays
#     branch_names = ['PFE High-Freq', 'PFE Low-Freq', 'SFE Semantic', 'Fused']
#     branch_keys = ['pfe_high', 'pfe_low', 'sfe', 'fused']
#     positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    
#     for (row, col), name, key in zip(positions, branch_names, branch_keys):
#         cam = cams[key]
        
#         # Resize CAM to match image size if needed
#         if cam.shape != (rgb_img.shape[0], rgb_img.shape[1]):
#             cam_resized = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]))
#         else:
#             cam_resized = cam
        
#         # Normalize CAM to [0, 1]
#         cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        
#         # Apply colormap (jet)
#         cam_colored = plt.cm.jet(cam_normalized)[:, :, :3]  # [H, W, 3], remove alpha channel
        
#         # Blend with original image
#         alpha = 0.5
#         cam_overlay = (1 - alpha) * rgb_img + alpha * cam_colored
#         cam_overlay = np.clip(cam_overlay, 0, 1)
        
#         axes[row, col].imshow(cam_overlay)
#         axes[row, col].set_title(name)
#         axes[row, col].axis('off')
    
#     # Raw fused heatmap
#     axes[1, 2].imshow(cams['fused'], cmap='jet')
#     axes[1, 2].set_title('Fused Heatmap (raw)')
#     axes[1, 2].axis('off')
    
#     plt.tight_layout()
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Clean filename
#     safe_filename = "".join([c for c in str(file_name) if c.isalnum() or c in (' ', '-', '_')]).rstrip()
#     if not safe_filename:
#         safe_filename = 'sample'
    
#     plt.savefig(os.path.join(output_dir, f'{safe_filename}_cam.png'), dpi=150, bbox_inches='tight')
#     plt.close()

def save_cam_visualization(rgb_image, cams, file_name, output_dir, pred_prob, true_label, pred_label):
    """
    Save CAM visualizations as a grid
    Args:
        rgb_image: [3, H, W] tensor
        cams: dict of {branch: heatmap [H, W]}
        file_name: str (now includes category prefix like "TP_", "FP_", etc.)
        output_dir: str
        pred_prob: float
        true_label: int
        pred_label: int
    """
    # Determine category from filename for title
    category = file_name.split('_')[0] if file_name.startswith(('TP', 'TN', 'FP', 'FN')) else ''
    category_full = {
        'TP': 'True Positive',
        'TN': 'True Negative', 
        'FP': 'False Positive',
        'FN': 'False Negative'
    }.get(category, '')
    
    # Convert RGB image to numpy [H, W, 3] in range [0, 1]
    rgb_img = rgb_image.permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    title_text = f'{file_name}\n{category_full}\nPred: {pred_label} ({pred_prob:.3f}) | True: {true_label}'
    fig.suptitle(title_text, fontsize=14, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # CAM overlays
    branch_names = ['PFE High-Freq', 'PFE Low-Freq', 'SFE Semantic', 'Fused']
    branch_keys = ['pfe_high', 'pfe_low', 'sfe', 'fused']
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    
    for (row, col), name, key in zip(positions, branch_names, branch_keys):
        cam = cams[key]
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]))
        # Create overlay
        cam_overlay = show_cam_on_image(rgb_img, cam_resized, use_rgb=True)
        axes[row, col].imshow(cam_overlay)
        axes[row, col].set_title(name)
        axes[row, col].axis('off')
    
    # Raw fused heatmap
    axes[1, 2].imshow(cams['fused'], cmap='jet')
    axes[1, 2].set_title('Fused Heatmap (raw)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{file_name}_cam.png'), dpi=150, bbox_inches='tight')
    plt.close()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        
        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# @torch.no_grad()
# # def evaluate(data_loader, model, device, use_amp=False):
# def evaluate(data_loader, model, device, use_amp=False, distributed=False):
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()

#     for index, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
#         images = batch[0]
#         target = batch[-1]

#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         if use_amp:
#             with torch.cuda.amp.autocast(dytpe=torch.bfloat16):
#                 output = model(images)
#                 if isinstance(output, dict):
#                     output = output['logits']
#                 loss = criterion(output, target)
#         else:
#             output = model(images) #[bs, num_cls]
#             if isinstance(output, dict):
#                 output = output['logits']
            
#             loss = criterion(output, target)
        
#         if index == 0:
#             predictions = output
#             labels = target
#         else:
#             predictions = torch.cat((predictions, output), 0)
#             labels = torch.cat((labels, target), 0)

#         torch.cuda.synchronize()

#         acc1, acc5 = accuracy(output, target, topk=(1, 2))

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

#     if distributed:
#         output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
#         dist.all_gather(output_ddp, predictions)
#         labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
#         dist.all_gather(labels_ddp, labels)

#         output_all = torch.concat(output_ddp, dim=0)
#         labels_all = torch.concat(labels_ddp, dim=0)
    
#     else:
#         output_all = predictions
#         labels_all = labels


#     y_pred = softmax(output_all.detach().cpu().numpy(), axis=1)[:, 1]
#     y_true = labels_all.detach().cpu().numpy()
#     y_true = y_true.astype(int)
    
  
#     acc = accuracy_score(y_true, y_pred > 0.5)
#     ap = average_precision_score(y_true, y_pred)
    
   

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, ap


# @torch.no_grad()
# def evaluate(data_loader, model, device, use_amp=False, distributed=False, 
#              generate_cams=False, cam_output_dir=None, num_cam_samples=100):
#     criterion = torch.nn.CrossEntropyLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     # switch to evaluation mode
#     model.eval()

#     for index, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
#         images = batch[0]
#         target = batch[-1]

#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         if use_amp:
#             with torch.cuda.amp.autocast(dytpe=torch.bfloat16):
#                 output = model(images)
#                 if isinstance(output, dict):
#                     output = output['logits']
#                 loss = criterion(output, target)
#         else:
#             output = model(images) #[bs, num_cls]
#             if isinstance(output, dict):
#                 output = output['logits']
            
#             loss = criterion(output, target)
        
#         if index == 0:
#             predictions = output
#             labels = target
#         else:
#             predictions = torch.cat((predictions, output), 0)
#             labels = torch.cat((labels, target), 0)

#         torch.cuda.synchronize()

#         acc1, acc5 = accuracy(output, target, topk=(1, 2))

#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
#     # my code
#     if distributed:
#         output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
#         dist.all_gather(output_ddp, predictions)
#         labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
#         dist.all_gather(labels_ddp, labels)

#         output_all = torch.concat(output_ddp, dim=0)
#         labels_all = torch.concat(labels_ddp, dim=0)

#     else:
#         output_all = predictions
#         labels_all = labels


#     output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
#     # dist.all_gather(output_ddp, predictions)
    
#     # my code for fixing error:
#     if utils.is_dist_avail_and_initialized():
#         dist.all_gather(output_ddp, predictions)
#     else:
#         output_ddp = [predictions]
#     # my code ends
    
#     labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
#     # dist.all_gather(labels_ddp, labels)
    
#     # my code to fix error
#     if utils.is_dist_avail_and_initialized():
#         dist.all_gather(labels_ddp, labels)
#     else:
#         labels_ddp = [labels]
#     # my code ends

#     output_all = torch.concat(output_ddp, dim=0)
#     labels_all = torch.concat(labels_ddp, dim=0)


#     y_pred = softmax(output_all.detach().cpu().numpy(), axis=1)[:, 1]
#     y_true = labels_all.detach().cpu().numpy()
#     y_true = y_true.astype(int)
    
#     # my code for printing predictions and ground truth 
#     # for i in range(0, len(y_true)):
#     #     print(y_pred[i], y_true[i])
  
#     acc = accuracy_score(y_true, y_pred > 0.5)
#     ap = average_precision_score(y_true, y_pred)
    
#     # NEW: Generate Grad-CAM++ visualizations if requested
#     cam_results = None
#     if generate_cams and cam_output_dir is not None:
#         print(f"\nGenerating Grad-CAM++ visualizations...")
#         os.makedirs(cam_output_dir, exist_ok=True)
        
#         cam_results = {
#             'pfe_high': [],
#             'pfe_low': [],
#             'sfe': [],
#             'fused': []
#         }
        
#         model.eval()
#         torch.set_grad_enabled(True)  # Enable gradients for CAM
        
#         cam_count = 0
#         for batch_idx, batch in enumerate(data_loader):
#             if cam_count >= num_cam_samples:
#                 break
            
#             images = batch[0].to(device)
#             target = batch[-1].to(device)
            
#             # Get file names if available
#             try:
#                 file_names = batch[2] if len(batch) > 2 else None
#             except:
#                 file_names = None
            
#             batch_size = images.shape[0]
            
#             for i in range(batch_size):
#                 if cam_count >= num_cam_samples:
#                     break
                
#                 single_image = images[i:i+1]  # [1, 5, 3, H, W]
#                 single_target = target[i].item()
                
#                 # Get prediction
#                 with torch.no_grad():
#                     logits = model(single_image)
#                     pred_class = logits.argmax(dim=1).item()
#                     pred_prob = torch.softmax(logits, dim=1)[0, 1].item()
                
#                 # Generate CAMs for all branches
#                 try:
#                     cams = model.module.generate_gradcam(
#                         single_image, 
#                         target_class=1,  # Always target "fake" class
#                         branches=['pfe_high', 'pfe_low', 'sfe']
#                     ) if hasattr(model, 'module') else model.generate_gradcam(
#                         single_image,
#                         target_class=1,
#                         branches=['pfe_high', 'pfe_low', 'sfe']
#                     )
                    
#                     # Fuse CAMs (simple average)
#                     fused_cam = np.mean([
#                         cams['pfe_high'],
#                         cams['pfe_low'],
#                         cams['sfe']
#                     ], axis=0)
#                     cams['fused'] = fused_cam
                    
#                     # Store results
#                     for branch in ['pfe_high', 'pfe_low', 'sfe', 'fused']:
#                         cam_results[branch].append(cams[branch])
                    
#                     # Save visualization
#                     file_name = file_names[i] if file_names else f"sample_{cam_count}"
#                     save_cam_visualization(
#                         single_image[0, 4].cpu(),  # Use the RGB token image
#                         cams,
#                         file_name,
#                         cam_output_dir,
#                         pred_prob,
#                         single_target,
#                         pred_class
#                     )
                    
#                     cam_count += 1
#                     if cam_count % 10 == 0:
#                         print(f"Generated {cam_count}/{num_cam_samples} CAMs")
                
#                 except Exception as e:
#                     print(f"Error generating CAM for sample {cam_count}: {e}")
#                     continue
        
#         torch.set_grad_enabled(False)  # Disable gradients again
#         print(f"✅ Generated {cam_count} Grad-CAM++ visualizations in {cam_output_dir}")
    
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, ap, y_pred, y_true, cam_results

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, ap
    
    # my code to return predictions and ground truths
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, ap, y_pred, y_true



@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, distributed=False, 
             generate_cams=False, cam_output_dir=None, num_cam_samples=100):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for index, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast(dytpe=torch.bfloat16):
                output = model(images)
                if isinstance(output, dict):
                    output = output['logits']
                loss = criterion(output, target)
        else:
            output = model(images) #[bs, num_cls]
            if isinstance(output, dict):
                output = output['logits']
            
            loss = criterion(output, target)
        
        if index == 0:
            predictions = output
            labels = target
        else:
            predictions = torch.cat((predictions, output), 0)
            labels = torch.cat((labels, target), 0)

        torch.cuda.synchronize()

        acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    # my code
    if distributed:
        output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
        dist.all_gather(output_ddp, predictions)
        labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
        dist.all_gather(labels_ddp, labels)

        output_all = torch.concat(output_ddp, dim=0)
        labels_all = torch.concat(labels_ddp, dim=0)

    else:
        output_all = predictions
        labels_all = labels


    output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
    
    if utils.is_dist_avail_and_initialized():
        dist.all_gather(output_ddp, predictions)
    else:
        output_ddp = [predictions]
    
    labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
    
    if utils.is_dist_avail_and_initialized():
        dist.all_gather(labels_ddp, labels)
    else:
        labels_ddp = [labels]

    output_all = torch.concat(output_ddp, dim=0)
    labels_all = torch.concat(labels_ddp, dim=0)


    y_pred = softmax(output_all.detach().cpu().numpy(), axis=1)[:, 1]
    y_true = labels_all.detach().cpu().numpy()
    y_true = y_true.astype(int)
  
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    
    # NEW: Generate Grad-CAM++ visualizations if requested
    cam_results = None
    if generate_cams and cam_output_dir is not None:
        print(f"\nGenerating Grad-CAM++ visualizations for TP/TN/FP/FN...")
        os.makedirs(cam_output_dir, exist_ok=True)
        
        cam_results = {
            'pfe_high': [],
            'pfe_low': [],
            'sfe': [],
            'fused': []
        }
        
        # Track counts for each category
        cam_counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        samples_per_category = num_cam_samples // 4  # Divide equally among 4 categories
        
        model.eval()
        torch.set_grad_enabled(True)  # Enable gradients for CAM
        
        total_cam_count = 0
        for batch_idx, batch in enumerate(data_loader):
            if total_cam_count >= num_cam_samples:
                break
            
            # Check if we've collected enough samples for all categories
            if all(count >= samples_per_category for count in cam_counts.values()):
                break
            
            images = batch[0].to(device)
            target = batch[-1].to(device)
            
            # Get file names if available
            try:
                file_names = batch[2] if len(batch) > 2 else None
            except:
                file_names = None
            
            batch_size = images.shape[0]
            
            for i in range(batch_size):
                if total_cam_count >= num_cam_samples:
                    break
                
                single_image = images[i:i+1]  # [1, 5, 3, H, W]
                single_target = target[i].item()
                
                # Get prediction
                with torch.no_grad():
                    logits = model(single_image)
                    pred_class = logits.argmax(dim=1).item()
                    pred_prob = torch.softmax(logits, dim=1)[0, 1].item()
                
                # Determine category: TP, TN, FP, or FN
                if single_target == 1 and pred_class == 1:
                    category = 'TP'
                elif single_target == 0 and pred_class == 0:
                    category = 'TN'
                elif single_target == 0 and pred_class == 1:
                    category = 'FP'
                elif single_target == 1 and pred_class == 0:
                    category = 'FN'
                
                # Skip if we have enough samples for this category
                if cam_counts[category] >= samples_per_category:
                    continue
                
                # Generate CAMs for all branches
                try:
                    cams = model.module.generate_gradcam(
                        single_image, 
                        target_class=1,  # Always target "fake" class
                        branches=['pfe_high', 'pfe_low', 'sfe']
                    ) if hasattr(model, 'module') else model.generate_gradcam(
                        single_image,
                        target_class=1,
                        branches=['pfe_high', 'pfe_low', 'sfe']
                    )
                    
                    # Fuse CAMs (simple average)
                    fused_cam = np.mean([
                        cams['pfe_high'],
                        cams['pfe_low'],
                        cams['sfe']
                    ], axis=0)
                    cams['fused'] = fused_cam
                    
                    # Store results
                    for branch in ['pfe_high', 'pfe_low', 'sfe', 'fused']:
                        cam_results[branch].append(cams[branch])
                    
                    # Save visualization with category label
                    file_name = file_names[i] if file_names else f"sample_{total_cam_count}"
                    file_name_with_category = f"{category}_{file_name}"
                    save_cam_visualization(
                        single_image[0, 4].cpu(),  # Use the RGB token image
                        cams,
                        file_name_with_category,
                        cam_output_dir,
                        pred_prob,
                        single_target,
                        pred_class
                    )
                    
                    cam_counts[category] += 1
                    total_cam_count += 1
                    
                    if total_cam_count % 10 == 0:
                        print(f"Generated {total_cam_count}/{num_cam_samples} CAMs - "
                              f"TP: {cam_counts['TP']}, TN: {cam_counts['TN']}, "
                              f"FP: {cam_counts['FP']}, FN: {cam_counts['FN']}")
                
                except Exception as e:
                    print(f"Error generating CAM for sample {total_cam_count}: {e}")
                    continue
        
        torch.set_grad_enabled(False)  # Disable gradients again
        print(f"✅ Generated {total_cam_count} Grad-CAM++ visualizations in {cam_output_dir}")
        print(f"   TP: {cam_counts['TP']}, TN: {cam_counts['TN']}, "
              f"FP: {cam_counts['FP']}, FN: {cam_counts['FN']}")
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, ap, y_pred, y_true, cam_results

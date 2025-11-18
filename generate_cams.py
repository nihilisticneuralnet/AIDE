"""
Grad-CAM++ Visualization Script for AIDE Model

This script generates and saves attribution heatmaps for the three branches:
- PFE-High: High-frequency patchwise features (model_min)
- PFE-Low: Low-frequency patchwise features (model_max)
- SFE: Semantic features from ConvNeXt

Usage:
    python generate_cams.py --checkpoint path/to/model.pth \
                           --data_dir path/to/images \
                           --output_dir ./cam_outputs \
                           --branches pfe_high pfe_low sfe \
                           --num_samples 100
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import your AIDE model
import models.AIDE as AIDE

class ImageDataset(Dataset):
    """Simple dataset for loading images"""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(list(self.image_dir.glob('*.jpg')) + 
                                  list(self.image_dir.glob('*.png')))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, str(img_path.name)


def prepare_aide_input(image_tensor, device):
    """
    Prepare input tensor for AIDE model.
    
    Args:
        image_tensor: [B, C, H, W] normalized image tensor
        device: torch device
    
    Returns:
        [B, 5, C, H, W] tensor with duplicated inputs for AIDE
    """
    b, c, h, w = image_tensor.shape
    
    # AIDE expects [B, T=5, C, H, W]
    # For visualization, we use the same image for all temporal positions
    x = image_tensor.unsqueeze(1).repeat(1, 5, 1, 1, 1)
    
    return x.to(device)


def upsample_heatmap(heatmap, target_size):
    """
    Upsample heatmap to target size using bilinear interpolation.
    
    Args:
        heatmap: [B, H, W] or [H, W] tensor
        target_size: (H, W) tuple
    
    Returns:
        Upsampled heatmap
    """
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(1)
    
    upsampled = F.interpolate(
        heatmap,
        size=target_size,
        mode='bilinear',
        align_corners=False
    )
    
    return upsampled.squeeze(1) if upsampled.dim() == 4 else upsampled.squeeze()


def apply_colormap(heatmap_np, colormap=cv2.COLORMAP_JET):
    """
    Apply colormap to heatmap.
    
    Args:
        heatmap_np: [H, W] numpy array in [0, 1]
        colormap: OpenCV colormap
    
    Returns:
        [H, W, 3] RGB image
    """
    heatmap_uint8 = (heatmap_np * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    return colored_heatmap


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay heatmap on image.
    
    Args:
        image: [H, W, 3] numpy array in [0, 255]
        heatmap: [H, W, 3] colored heatmap in [0, 255]
        alpha: Overlay transparency
    
    Returns:
        [H, W, 3] overlaid image
    """
    return (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)


def denormalize_image(tensor, mean, std):
    """Denormalize image tensor to [0, 1] range"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def save_cam_visualization(
    original_image,
    heatmap,
    save_path,
    title='Grad-CAM++',
    alpha=0.5
):
    """
    Save CAM visualization with original image, heatmap, and overlay.
    
    Args:
        original_image: [C, H, W] tensor or [H, W, C] numpy array
        heatmap: [H, W] tensor or numpy array
        save_path: Output file path
        title: Title for the plot
        alpha: Overlay alpha
    """
    # Convert to numpy if needed
    if isinstance(original_image, torch.Tensor):
        if original_image.dim() == 3 and original_image.shape[0] == 3:
            original_image = original_image.permute(1, 2, 0)
        original_image = original_image.cpu().numpy()
    
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # Ensure original image is in [0, 255]
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    else:
        original_image = original_image.astype(np.uint8)
    
    # Ensure heatmap is in [0, 1]
    if heatmap.max() > 1.0:
        heatmap = heatmap / 255.0
    
    # Resize heatmap to match image
    if heatmap.shape != original_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Apply colormap
    colored_heatmap = apply_colormap(heatmap)
    
    # Create overlay
    overlaid = overlay_heatmap(original_image, colored_heatmap, alpha)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(colored_heatmap)
    axes[1].set_title(f'{title} Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlaid)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_fused_heatmap(heatmaps_dict, weights=None):
    """
    Generate fused heatmap from multiple branches.
    
    Args:
        heatmaps_dict: Dict mapping branch name to heatmap tensor
        weights: Dict mapping branch name to weight (default: equal weights)
    
    Returns:
        Fused heatmap tensor
    """
    if weights is None:
        weights = {k: 1.0 / len(heatmaps_dict) for k in heatmaps_dict.keys()}
    
    fused = None
    for branch, heatmap in heatmaps_dict.items():
        weighted = heatmap * weights.get(branch, 1.0 / len(heatmaps_dict))
        if fused is None:
            fused = weighted
        else:
            fused += weighted
    
    # Normalize to [0, 1]
    fused_min = fused.min()
    fused_max = fused.max()
    if fused_max - fused_min > 1e-10:
        fused = (fused - fused_min) / (fused_max - fused_min)
    
    return fused


def generate_cams_for_dataset(
    model,
    dataloader,
    device,
    output_dir,
    branches=['pfe_high', 'pfe_low', 'sfe'],
    save_raw=True,
    save_overlay=True,
    save_fused=True,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225)
):
    """
    Generate Grad-CAM++ heatmaps for entire dataset.
    
    Args:
        model: AIDE model
        dataloader: DataLoader for images
        device: torch device
        output_dir: Output directory
        branches: List of branches to compute CAMs for
        save_raw: Save raw heatmap arrays
        save_overlay: Save overlay visualizations
        save_fused: Save fused heatmap from all branches
        image_mean: Mean for denormalization
        image_std: Std for denormalization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for branch in branches:
        (output_dir / branch).mkdir(exist_ok=True)
        if save_raw:
            (output_dir / branch / 'raw').mkdir(exist_ok=True)
        if save_overlay:
            (output_dir / branch / 'overlay').mkdir(exist_ok=True)
    
    if save_fused and len(branches) > 1:
        (output_dir / 'fused').mkdir(exist_ok=True)
        if save_raw:
            (output_dir / 'fused' / 'raw').mkdir(exist_ok=True)
        if save_overlay:
            (output_dir / 'fused' / 'overlay').mkdir(exist_ok=True)
    
    model.eval()
    
    print(f"Generating Grad-CAM++ visualizations for {len(branches)} branches...")
    
    for batch_idx, (images, filenames) in enumerate(tqdm(dataloader)):
        # Move images to device - at this point images is [B, C, H, W]
        images = images.to(device)
        b, c, h, w = images.shape
        
        # Prepare AIDE input - this converts [B, C, H, W] to [B, 5, C, H, W]
        aide_input = prepare_aide_input(images, device)
        
        # Store heatmaps for fusion
        all_heatmaps = {branch: [] for branch in branches}
        
        # Generate CAMs for each branch
        for branch in branches:
            with torch.set_grad_enabled(True):
                # Forward pass with CAM
                logits, cam_heatmap = model(aide_input, return_cam=True, cam_layer=branch)
                
                # Upsample to original image size
                cam_upsampled = upsample_heatmap(cam_heatmap, (h, w))
                
                all_heatmaps[branch] = cam_upsampled
                
                # Save per-sample
                for i in range(b):
                    filename = Path(filenames[i]).stem
                    
                    # Get original image for visualization
                    orig_img = denormalize_image(
                        images[i],
                        mean=image_mean,
                        std=image_std
                    )
                    
                    heatmap_np = cam_upsampled[i].detach().cpu().numpy()
                    
                    # Save raw heatmap
                    if save_raw:
                        raw_path = output_dir / branch / 'raw' / f'{filename}.npy'
                        np.save(raw_path, heatmap_np)
                    
                    # Save overlay visualization
                    if save_overlay:
                        overlay_path = output_dir / branch / 'overlay' / f'{filename}.png'
                        save_cam_visualization(
                            orig_img,
                            heatmap_np,
                            overlay_path,
                            title=f'{branch.upper()} Grad-CAM++'
                        )
            
            # Clean up hooks after each branch
            model.cleanup_gradcam(branch)
        
        # Generate and save fused heatmap
        if save_fused and len(branches) > 1:
            for i in range(b):
                filename = Path(filenames[i]).stem
                
                # Collect heatmaps for this sample
                sample_heatmaps = {
                    branch: all_heatmaps[branch][i]
                    for branch in branches
                }
                
                # Fuse heatmaps
                fused_heatmap = generate_fused_heatmap(sample_heatmaps)
                fused_np = fused_heatmap.detach().cpu().numpy()
                
                # Get original image
                orig_img = denormalize_image(
                    images[i],
                    mean=image_mean,
                    std=image_std
                )
                
                # Save raw fused heatmap
                if save_raw:
                    raw_path = output_dir / 'fused' / 'raw' / f'{filename}.npy'
                    np.save(raw_path, fused_np)
                
                # Save fused overlay
                if save_overlay:
                    overlay_path = output_dir / 'fused' / 'overlay' / f'{filename}.png'
                    save_cam_visualization(
                        orig_img,
                        fused_np,
                        overlay_path,
                        title='Fused Grad-CAM++'
                    )
    
    print(f"\nCAM generation complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM++ visualizations for AIDE model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--resnet_path', type=str, required=True,
                        help='Path to ResNet pretrained weights')
    parser.add_argument('--convnext_path', type=str, required=True,
                        help='Path to ConvNeXt pretrained weights')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='./cam_outputs',
                        help='Output directory for CAM visualizations')
    parser.add_argument('--branches', nargs='+', 
                        default=['pfe_high', 'pfe_low', 'sfe'],
                        choices=['pfe_high', 'pfe_low', 'sfe'],
                        help='Branches to compute CAMs for')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (None for all)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--no_raw', action='store_true',
                        help='Do not save raw heatmap arrays')
    parser.add_argument('--no_overlay', action='store_true',
                        help='Do not save overlay visualizations')
    parser.add_argument('--no_fused', action='store_true',
                        help='Do not save fused heatmaps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading AIDE model...")
    model = AIDE.AIDE(args.resnet_path, args.convnext_path)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Setup data transforms
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create dataset
    dataset = ImageDataset(args.data_dir, transform=transform)
    
    if args.num_samples is not None:
        dataset.image_paths = dataset.image_paths[:args.num_samples]
    
    print(f"Processing {len(dataset)} images...")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Generate CAMs
    generate_cams_for_dataset(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=args.output_dir,
        branches=args.branches,
        save_raw=not args.no_raw,
        save_overlay=not args.no_overlay,
        save_fused=not args.no_fused
    )
    
    # Cleanup
    model.cleanup_gradcam()


if __name__ == '__main__':
    main()

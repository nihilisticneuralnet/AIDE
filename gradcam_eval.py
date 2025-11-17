"""
Grad-CAM++ Evaluation Script for AIDE
This script can be run separately or integrated with main.py

Usage:
    python gradcam_eval.py --model_path weight/w1.pth \
                           --resnet_path resnet.pth \
                           --convnext_path weight/w2.pth \
                           --eval_data_path path/to/dataset \
                           --output_dir gradcam_output \
                           --batch_size 8 \
                           --max_samples 100
"""

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path

# Import from your codebase
from data.datasets import TestDataset
import models.AIDE as AIDE
from models.AIDE import GradCAMAnalyzer
import utils


def get_args_parser():
    parser = argparse.ArgumentParser('AIDE Grad-CAM++ Evaluation', add_help=False)
    
    # Model parameters
    parser.add_argument('--model_path', required=True, type=str,
                        help='Path to trained model checkpoint (e.g., weight/w1.pth)')
    parser.add_argument('--resnet_path', required=True, type=str,
                        help='Path to ResNet pretrained weights')
    parser.add_argument('--convnext_path', required=True, type=str,
                        help='Path to ConvNeXt pretrained weights')
    
    # Data parameters
    parser.add_argument('--eval_data_path', required=True, type=str,
                        help='Path to evaluation dataset')
    parser.add_argument('--data_path', default='', type=str,
                        help='Base data path (for compatibility)')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    
    # Grad-CAM parameters
    parser.add_argument('--output_dir', default='gradcam_output', type=str,
                        help='Directory to save Grad-CAM visualizations')
    parser.add_argument('--max_samples', default=100, type=int,
                        help='Maximum number of samples to analyze')
    parser.add_argument('--save_all', action='store_true',
                        help='Save Grad-CAMs for all samples (not just FP/FN)')
    parser.add_argument('--branches', default='all', choices=['all', 'patchwise', 'semantic'],
                        help='Which branches to visualize')
    
    # System parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--imagenet_default_mean_and_std', type=bool, default=True)
    
    return parser


def load_model(args):
    """Load AIDE model with trained weights"""
    print(f"Loading model from: {args.model_path}")
    
    # Create model
    model = AIDE.__dict__['AIDE'](
        resnet_path=args.resnet_path,
        convnext_path=args.convnext_path
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(args.device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def evaluate_with_gradcam(args):
    """Main evaluation function with Grad-CAM++ visualization"""
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    
    # Load model
    model = load_model(args)
    
    # Create Grad-CAM analyzer
    analyzer = GradCAMAnalyzer(model, device=args.device)
    
    # Load dataset
    print(f"Loading dataset from: {args.eval_data_path}")
    dataset = TestDataset(is_train=False, args=args)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze dataset with Grad-CAM++
    print("\n" + "="*80)
    print("Starting Grad-CAM++ Analysis")
    print("="*80)
    
    stats = analyzer.analyze_dataset(
        data_loader=data_loader,
        output_root=args.output_dir,
        max_samples=args.max_samples,
        save_fp_fn=True
    )
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Accuracy: {stats['accuracy']*100:.2f}%")
    print(f"False Positives: {stats['fp']}")
    print(f"False Negatives: {stats['fn']}")
    
    # Optional: Generate visualizations for random samples
    if args.save_all:
        print("\nGenerating visualizations for random samples...")
        random_dir = os.path.join(args.output_dir, 'random_samples')
        
        # Get a few random samples
        indices = np.random.choice(len(dataset), min(20, len(dataset)), replace=False)
        
        for idx in indices:
            sample = dataset[idx]
            if isinstance(sample, tuple):
                image, label = sample
            else:
                image = sample
                label = None
            
            image = image.unsqueeze(0).to(args.device)
            
            analyzer.visualize_all_branches(
                image,
                random_dir,
                f'sample_{idx}',
                true_label=label
            )
    
    return stats


def main():
    parser = argparse.ArgumentParser('AIDE Grad-CAM++ Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    stats = evaluate_with_gradcam(args)
    
    print("\nDone!")


if __name__ == '__main__':
    main()


"""
EXAMPLE COMMANDS:

1. Basic Grad-CAM++ evaluation:
    python gradcam_eval.py \
        --model_path weight/w1.pth \
        --resnet_path resnet.pth \
        --convnext_path weight/w2.pth \
        --eval_data_path path/to/test/dataset \
        --output_dir gradcam_results \
        --max_samples 100

2. Evaluate with all visualizations:
    python gradcam_eval.py \
        --model_path weight/w1.pth \
        --resnet_path resnet.pth \
        --convnext_path weight/w2.pth \
        --eval_data_path path/to/test/dataset \
        --output_dir gradcam_results \
        --max_samples 200 \
        --save_all

3. Evaluate on specific dataset subset:
    python gradcam_eval.py \
        --model_path weight/w1.pth \
        --resnet_path resnet.pth \
        --convnext_path weight/w2.pth \
        --eval_data_path path/to/test/dataset/Midjourney \
        --output_dir gradcam_results/midjourney \
        --max_samples 50

OUTPUT STRUCTURE:
    gradcam_results/
    ├── false_positives/
    │   ├── FP_0_batch0_img0_patchwise_minmin.png
    │   ├── FP_0_batch0_img0_patchwise_maxmax.png
    │   ├── FP_0_batch0_img0_patchwise_minmin1.png
    │   ├── FP_0_batch0_img0_patchwise_maxmax1.png
    │   ├── FP_0_batch0_img0_semantic.png
    │   └── FP_0_batch0_img0_original.png
    ├── false_negatives/
    │   └── [similar structure]
    ├── random_samples/ (if --save_all is used)
    │   └── [similar structure]
    └── analysis_stats.json
"""
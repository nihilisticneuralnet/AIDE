import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import clip
import open_clip
from .srm_filter_kernel import all_normalized_hpf_list
import numpy as np
import os
from PIL import Image
import cv2
from tqdm import tqdm

from typing import Dict, List, Optional, Tuple


class GradCAMPlusPlus:
    """
    GradCAM++ implementation for visualizing model attention.
    Supports multiple target layers and handles mixture-of-experts architectures.
    """
    
    def __init__(self, model: nn.Module, target_layers: Dict[str, nn.Module]):
        """
        Args:
            model: The neural network model
            target_layers: Dictionary mapping names to target layers
                          e.g., {'pfe_min': model.model_min.layer4[-1]}
        """
        self.model = model
        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
        # Register hooks for each target layer
        for name, layer in target_layers.items():
            self.hooks.append(
                layer.register_forward_hook(self._save_activation(name))
            )
            self.hooks.append(
                layer.register_full_backward_hook(self._save_gradient(name))
            )
    
    def _save_activation(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def _save_gradient(self, name: str):
        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()
        return hook
    
    def compute_cam(self, target_class: int = None) -> Dict[str, torch.Tensor]:
        """
        Compute GradCAM++ for all registered target layers.
        
        Args:
            target_class: Class index to compute gradients for. 
                         If None, uses the predicted class.
        
        Returns:
            Dictionary mapping layer names to CAM heatmaps [B, H, W]
        """
        cams = {}
        
        for name in self.target_layers.keys():
            if name not in self.activations or name not in self.gradients:
                print(f"Warning: No activations/gradients found for {name}")
                continue
            
            activations = self.activations[name]  # [B, C, H, W]
            gradients = self.gradients[name]      # [B, C, H, W]
            
            b, c, h, w = activations.shape
            
            # Compute alpha weights using GradCAM++ formulation
            alpha_numer = gradients.pow(2)
            alpha_denom = 2 * gradients.pow(2) + \
                         (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True)
            alpha_denom = alpha_denom + 1e-8  # Numerical stability
            
            alpha = alpha_numer / alpha_denom
            
            # Apply ReLU to gradients (positive influence)
            relu_grad = F.relu(gradients)
            
            # Compute weights
            weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            
            # Weighted combination of activation maps
            cam = (weights * activations).sum(dim=1)  # [B, H, W]
            
            # Apply ReLU to CAM
            cam = F.relu(cam)
            
            # Normalize CAM to [0, 1]
            cam = self._normalize_cam(cam)
            
            cams[name] = cam
        
        return cams
    
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        """Normalize CAM to [0, 1] range per sample."""
        b = cam.shape[0]
        cam = cam.view(b, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.view(b, *cam.shape[1:])
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}


def upsample_cam(cam: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """
    Upsample CAM to target size using bilinear interpolation.
    
    Args:
        cam: CAM tensor [B, H, W]
        target_size: (height, width)
    
    Returns:
        Upsampled CAM [B, H_target, W_target]
    """
    if len(cam.shape) == 2:
        cam = cam.unsqueeze(0)
    cam = cam.unsqueeze(1)  # [B, 1, H, W]
    upsampled = F.interpolate(
        cam, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    return upsampled.squeeze(1)  # [B, H_target, W_target]


def overlay_cam_on_image(
    image: np.ndarray, 
    cam: np.ndarray, 
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay CAM heatmap on original image.
    
    Args:
        image: Original image [H, W, 3] in range [0, 255]
        cam: CAM heatmap [H, W] in range [0, 1]
        alpha: Blending factor for overlay
        colormap: OpenCV colormap
    
    Returns:
        Overlayed image [H, W, 3]
    """
    # Convert CAM to heatmap
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Overlay
    overlayed = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    return overlayed


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5).contiguous()
        hpf_weight = torch.nn.Parameter(hpf_weight.repeat(1, 3, 1, 1), requires_grad=False)

        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

    def forward(self, input):
        output = self.hpf(input)
        return output


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class AIDE_Model(nn.Module):

    def __init__(self, resnet_path, convnext_path):
        super(AIDE_Model, self).__init__()
        self.hpf = HPF()
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3])
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3])
       
        if resnet_path is not None:
            pretrained_dict = torch.load(resnet_path, map_location='cpu', weights_only=False)
        
            model_min_dict = self.model_min.state_dict()
            model_max_dict = self.model_max.state_dict()
    
            for k in pretrained_dict.keys():
                if k in model_min_dict and pretrained_dict[k].size() == model_min_dict[k].size():
                    model_min_dict[k] = pretrained_dict[k]
                    model_max_dict[k] = pretrained_dict[k]
                else:
                    print(f"Skipping layer {k} because of size mismatch")
        
        self.fc = Mlp(2048 + 256, 1024, 2)

        print("build model with convnext_xxl")
        self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained=convnext_path
        )

        self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
        self.openclip_convnext_xxl.head.global_pool = nn.Identity()
        self.openclip_convnext_xxl.head.flatten = nn.Identity()

        self.openclip_convnext_xxl.eval()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnext_proj = nn.Sequential(
            nn.Linear(3072, 256),
        )
        
        for param in self.openclip_convnext_xxl.parameters():
            param.requires_grad = False
        
        # GradCAM++ support
        self.gradcam = None

    def setup_gradcam(self):
        """
        Initialize GradCAM++ with target layers from all three branches.
        
        Target layers:
        - pfe_min: Last conv block of lowest-frequency ResNet expert
        - pfe_max: Last conv block of highest-frequency ResNet expert  
        - sfe: Final ConvNeXt stage block
        """
        target_layers = {
            'pfe_min': self.model_min.layer4[-1],  # Last bottleneck of ResNet
            'pfe_max': self.model_max.layer4[-1],  # Last bottleneck of ResNet
        }
        
        # Find the last ConvNeXt block
        try:
            # ConvNeXt structure: stages -> blocks
            last_stage = self.openclip_convnext_xxl.stages[-1]
            if hasattr(last_stage, 'blocks'):
                target_layers['sfe'] = last_stage.blocks[-1]
            else:
                # Alternative: use the last stage itself
                target_layers['sfe'] = last_stage
        except Exception as e:
            print(f"Warning: Could not find ConvNeXt target layer: {e}")
        
        self.gradcam = GradCAMPlusPlus(self, target_layers)
        return self.gradcam

    def forward(self, x, return_features=False):
        """
        Forward pass with optional feature retention for GradCAM++.
        
        Args:
            x: Input tensor [B, T, C, H, W] where T contains different patches
            return_features: If True, keeps computation graph for gradients
        
        Returns:
            If return_features=False: logits [B, 2]
            If return_features=True: (logits, features_dict)
        """
        b, t, c, h, w = x.shape

        x_minmin = x[:, 0]   # [b, c, h, w] - lowest frequency patch
        x_maxmax = x[:, 1]   # highest frequency patch
        x_minmin1 = x[:, 2]  # lowest frequency patch (second)
        x_maxmax1 = x[:, 3]  # highest frequency patch (second)
        tokens = x[:, 4]     # semantic input

        # Apply HPF filters to all PFE inputs
        x_minmin_hpf = self.hpf(x_minmin)
        x_maxmax_hpf = self.hpf(x_maxmax)
        x_minmin1_hpf = self.hpf(x_minmin1)
        x_maxmax1_hpf = self.hpf(x_maxmax1)

        # Semantic Feature Embedding (SFE) branch
        if return_features:
            # Keep gradients for SFE when computing GradCAM
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

            # Enable gradients for ConvNeXt when visualizing
            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            )
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            x_0 = self.convnext_proj(local_convnext_image_feats)
        else:
            # Normal forward pass with no_grad for efficiency
            with torch.no_grad():
                clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
                clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
                clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
                clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
                dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
                dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

                local_convnext_image_feats = self.openclip_convnext_xxl(
                    tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
                )
                assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
                local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
                x_0 = self.convnext_proj(local_convnext_image_feats)

        # Patchwise Feature Extraction (PFE) branch
        x_min = self.model_min(x_minmin_hpf)
        x_max = self.model_max(x_maxmax_hpf)
        x_min1 = self.model_min(x_minmin1_hpf)
        x_max1 = self.model_max(x_maxmax1_hpf)

        x_1 = (x_min + x_max + x_min1 + x_max1) / 4
        x_0 *= 0  # As per original code

        # Concatenate features
        x = torch.cat([x_0, x_1], dim=1)

        # Final classification
        logits = self.fc(x)

        if return_features:
            features = {
                'x_0': x_0,  # Semantic features
                'x_1': x_1,  # Patchwise features
                'x_min': x_min,
                'x_max': x_max,
                'x_min1': x_min1,
                'x_max1': x_max1,
            }
            return logits, features
        
        return logits
    
    def compute_gradcam(
        self, 
        x: torch.Tensor, 
        target_class: Optional[int] = None,
        branch: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GradCAM++ attribution maps for the model.
        
        Args:
            x: Input tensor [B, T, C, H, W]
            target_class: Class to compute gradients for (0=fake, 1=real)
                         If None, uses predicted class
            branch: Specific branch to visualize ('pfe_min', 'pfe_max', 'sfe')
                   If None, computes all branches
        
        Returns:
            Dictionary with CAM heatmaps and metadata
        """
        if self.gradcam is None:
            self.setup_gradcam()
        
        # Enable gradient computation
        self.eval()
        x.requires_grad = True
        
        # Forward pass with feature retention
        logits, features = self.forward(x, return_features=True)
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1)
        
        # Zero gradients
        self.zero_grad()
        
        # Backward pass for target class
        if isinstance(target_class, int):
            target = logits[:, target_class].sum()
        else:
            target = logits[range(len(target_class)), target_class].sum()
        
        target.backward()
        
        # Compute CAMs
        cams = self.gradcam.compute_cam(target_class)
        
        # Filter by branch if specified
        if branch is not None:
            cams = {k: v for k, v in cams.items() if k == branch}
        
        return {
            'cams': cams,
            'logits': logits.detach(),
            'features': {k: v.detach() for k, v in features.items()},
            'target_class': target_class
        }
    
    def visualize_gradcam(
        self,
        x: torch.Tensor,
        original_images: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Generate GradCAM++ visualizations overlayed on original images.
        
        Args:
            x: Input tensor [B, T, C, H, W]
            original_images: Original images [B, H, W, 3] in range [0, 255]
            target_class: Class to visualize (0=fake, 1=real)
            alpha: Blending factor for overlay
        
        Returns:
            Dictionary mapping branch names to overlayed images [B, H, W, 3]
        """
        result = self.compute_gradcam(x, target_class)
        cams = result['cams']
        
        b, _, _, h_orig, w_orig = x.shape
        visualizations = {}
        
        for branch_name, cam in cams.items():
            # Upsample CAM to original resolution
            cam_upsampled = upsample_cam(cam, (h_orig, w_orig))
            
            # Overlay on each image in batch
            overlayed_batch = []
            for i in range(b):
                cam_np = cam_upsampled[i].cpu().numpy()
                img_np = original_images[i]
                overlayed = overlay_cam_on_image(img_np, cam_np, alpha)
                overlayed_batch.append(overlayed)
            
            visualizations[branch_name] = np.stack(overlayed_batch)
        
        return visualizations


def AIDE(resnet_path, convnext_path):
    model = AIDE_Model(resnet_path, convnext_path)
    return model



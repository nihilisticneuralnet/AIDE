import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import clip
import open_clip
from .srm_filter_kernel import all_normalized_hpf_list
import numpy as np
from typing import Optional, Dict, Tuple, List
import cv2


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for generating spatial attribution heatmaps.
    Uses first-, second-, and third-order gradients for improved localization.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: The neural network model
            target_layer: The layer to compute Grad-CAM++ on (typically last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_cam(self, class_idx: Optional[int] = None) -> torch.Tensor:
        """
        Compute Grad-CAM++ heatmap using α^k weights from first-, second-, and third-order gradients.
        
        Args:
            class_idx: Target class index. If None, uses predicted class.
        
        Returns:
            Normalized heatmap tensor of shape [B, H, W]
        """
        if self.gradients is None or self.activations is None:
            raise RuntimeError("No gradients captured. Run forward and backward pass first.")
        
        # Get dimensions
        b, k, h, w = self.activations.shape
        
        # Compute Grad-CAM++ weights using first, second, and third order gradients
        # First order: dY/dA
        first_derivative = self.gradients  # [B, K, H, W]
        
        # Second order: d²Y/dA²
        second_derivative = first_derivative.pow(2)
        
        # Third order: d³Y/dA³
        third_derivative = first_derivative.pow(3)
        
        # Global sum for normalization
        global_sum = self.activations.sum(dim=(2, 3), keepdim=True)  # [B, K, 1, 1]
        
        # Compute alpha weights (Equation 7 from Grad-CAM++ paper)
        # α^k_ij = (d²Y/dA²_ij) / (2 * d²Y/dA²_ij + Σ(A^k) * d³Y/dA³_ij)
        alpha_denom = 2 * second_derivative + (global_sum * third_derivative + 1e-10)
        alpha_num = second_derivative
        alpha = alpha_num / alpha_denom  # [B, K, H, W]
        
        # Apply ReLU to gradients (only positive influence)
        positive_gradients = F.relu(first_derivative)
        
        # Weight the gradients with alpha
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)  # [B, K, 1, 1]
        
        # Compute weighted combination of activations
        cam = (weights * self.activations).sum(dim=1)  # [B, H, W]
        
        # Apply ReLU to final CAM
        cam = F.relu(cam)
        
        # Normalize to [0, 1] per sample
        for i in range(b):
            cam_min = cam[i].min()
            cam_max = cam[i].max()
            if cam_max - cam_min > 1e-10:
                cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)
        
        return cam
    
    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            input_tensor: Input tensor to the model
            class_idx: Target class index
        
        Returns:
            Heatmap tensor normalized to [0, 1]
        """
        return self.compute_cam(class_idx)


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
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

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
    """
    AIDE Model with Grad-CAM++ Attribution Support
    
    Supports three attribution targets:
    - 'pfe_high': High-frequency ResNet branch (model_min on x_minmin, x_minmin1)
    - 'pfe_low': Low-frequency ResNet branch (model_max on x_maxmax, x_maxmax1)
    - 'sfe': Semantic Feature Embedding (ConvNeXt backbone)
    """

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
        
        # Grad-CAM++ objects (initialized on demand)
        self.gradcam_objects: Dict[str, GradCAMPlusPlus] = {}

    def get_target_layer(self, cam_layer: str) -> nn.Module:
        """
        Get the target layer for Grad-CAM++ based on branch.
        
        Args:
            cam_layer: One of 'pfe_high', 'pfe_low', 'sfe'
        
        Returns:
            Target convolutional layer
        """
        if cam_layer == 'pfe_high':
            # Last conv3 in layer4 of model_min (high-frequency branch)
            return self.model_min.layer4[-1].conv3
        elif cam_layer == 'pfe_low':
            # Last conv3 in layer4 of model_max (low-frequency branch)
            return self.model_max.layer4[-1].conv3
        elif cam_layer == 'sfe':
            # Last convolutional block in ConvNeXt before projection
            # ConvNeXt stages: self.openclip_convnext_xxl.stages
            return self.openclip_convnext_xxl.stages[-1].blocks[-1].conv_dw
        else:
            raise ValueError(f"Unknown cam_layer: {cam_layer}. Choose from 'pfe_high', 'pfe_low', 'sfe'")

    def setup_gradcam(self, cam_layer: str):
        """
        Initialize Grad-CAM++ for specified layer.
        
        Args:
            cam_layer: One of 'pfe_high', 'pfe_low', 'sfe'
        """
        if cam_layer not in self.gradcam_objects:
            target_layer = self.get_target_layer(cam_layer)
            self.gradcam_objects[cam_layer] = GradCAMPlusPlus(self, target_layer)

    def cleanup_gradcam(self, cam_layer: Optional[str] = None):
        """
        Remove Grad-CAM++ hooks.
        
        Args:
            cam_layer: Specific layer to clean up. If None, cleans all.
        """
        if cam_layer is None:
            for gc in self.gradcam_objects.values():
                gc.remove_hooks()
            self.gradcam_objects = {}
        elif cam_layer in self.gradcam_objects:
            self.gradcam_objects[cam_layer].remove_hooks()
            del self.gradcam_objects[cam_layer]

    def forward(self, x, return_cam: bool = False, cam_layer: Optional[str] = None):
        """
        Forward pass with optional Grad-CAM++ attribution.
        
        Args:
            x: Input tensor [B, T, C, H, W] where T=5
               T=0: x_minmin (high-freq min)
               T=1: x_maxmax (low-freq max)
               T=2: x_minmin1 (high-freq min alt)
               T=3: x_maxmax1 (low-freq max alt)
               T=4: tokens (original image for semantic features)
            return_cam: If True, compute and return Grad-CAM++ heatmap
            cam_layer: Target layer for Grad-CAM++: 'pfe_high', 'pfe_low', 'sfe'
        
        Returns:
            If return_cam is False:
                logits: [B, 2] classification logits
            If return_cam is True:
                (logits, cam_heatmap): logits and [B, H_cam, W_cam] heatmap
        """
        b, t, c, h, w = x.shape

        x_minmin = x[:, 0]  # [b, c, h, w]
        x_maxmax = x[:, 1]
        x_minmin1 = x[:, 2]
        x_maxmax1 = x[:, 3]
        tokens = x[:, 4]

        # Apply HPF to patchwise inputs
        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        # Semantic Feature Embedding (SFE)
        with torch.set_grad_enabled(return_cam and cam_layer == 'sfe'):
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            )  # [b, 3072, 8, 8]
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            x_0 = self.convnext_proj(local_convnext_image_feats)

        # Patchwise Feature Extraction (PFE)
        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)

        x_1 = (x_min + x_max + x_min1 + x_max1) / 4
        x_0 = x_0 * 0  # As in original code

        x = torch.cat([x_0, x_1], dim=1)
        logits = self.fc(x)

        if not return_cam:
            return logits

        # Compute Grad-CAM++
        if cam_layer is None:
            raise ValueError("cam_layer must be specified when return_cam=True")

        # Setup Grad-CAM++ if not already done
        self.setup_gradcam(cam_layer)

        # Backward pass on predicted class (or class 1 for fake detection)
        pred_class = logits.argmax(dim=1)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, pred_class.unsqueeze(1), 1.0)
        
        # Compute gradients
        self.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)

        # Generate heatmap
        cam_heatmap = self.gradcam_objects[cam_layer].compute_cam()

        return logits, cam_heatmap


def AIDE(resnet_path, convnext_path):
    model = AIDE_Model(resnet_path, convnext_path)
    return model


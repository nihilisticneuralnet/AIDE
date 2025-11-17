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

class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    #Load 30 SRM Filters
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

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
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


# ============================================================================
# GRADCAM++ WRAPPER CLASS
# ============================================================================
class AIDEGradCAMWrapper(nn.Module):
    """
    Wrapper to make AIDE compatible with Grad-CAM++
    Usage:
        wrapper = AIDEGradCAMWrapper(aide_model, branch='patchwise', sublayer_idx=0)
        target_layers = wrapper.get_target_layers()
        cam = GradCAMPlusPlus(model=wrapper, target_layers=target_layers)
    """
    def __init__(self, aide_model, branch='patchwise', sublayer_idx=0):
        super().__init__()
        self.aide_model = aide_model
        self.branch = branch
        self.sublayer_idx = sublayer_idx  # 0=minmin, 1=maxmax, 2=minmin1, 3=maxmax1
        
    def get_target_layers(self):
        """
        Returns the target layers for Grad-CAM++
        """
        if self.branch == 'patchwise':
            # Target the last bottleneck block of layer4 in ResNet
            return [self.aide_model.model_min.layer4[-1]]
        elif self.branch == 'semantic':
            # Target the last block in the last stage of ConvNeXt
            return [self.aide_model.openclip_convnext_xxl.stages[-1].blocks[-1]]
        else:
            raise ValueError(f"Unknown branch: {self.branch}. Use 'patchwise' or 'semantic'")
        
    def forward(self, x):
        """
        Forward pass for Grad-CAM++
        Routes through the appropriate branch
        """
        b, t, c, h, w = x.shape
        print(f"Input requires_grad: {x.requires_grad}")
        
        if self.branch == 'patchwise':
            # Extract all sublayers
            x_minmin = x[:, 0]   # [b, c, h, w]
            x_maxmax = x[:, 1]
            x_minmin1 = x[:, 2]
            x_maxmax1 = x[:, 3]
            tokens = x[:, 4]
            
            # Apply HPF to all sublayers
            x_minmin = self.aide_model.hpf(x_minmin)
            x_maxmax = self.aide_model.hpf(x_maxmax)
            x_minmin1 = self.aide_model.hpf(x_minmin1)
            x_maxmax1 = self.aide_model.hpf(x_maxmax1)
            
            # Process the TARGET sublayer (the one we want to visualize) through full ResNet
            # This is where Grad-CAM++ will hook the gradients
            sublayers = [x_minmin, x_maxmax, x_minmin1, x_maxmax1]
            x_target = sublayers[self.sublayer_idx]
            
            # Forward through ResNet for the TARGET sublayer (with gradient tracking)
            x_out = self.aide_model.model_min.conv1(x_target)
            x_out = self.aide_model.model_min.bn1(x_out)
            x_out = self.aide_model.model_min.relu(x_out)
            x_out = self.aide_model.model_min.maxpool(x_out)
            
            x_out = self.aide_model.model_min.layer1(x_out)
            x_out = self.aide_model.model_min.layer2(x_out)
            x_out = self.aide_model.model_min.layer3(x_out)
            x_out = self.aide_model.model_min.layer4(x_out)  # This is where CAM will hook!
            
            x_out = self.aide_model.model_min.avgpool(x_out)
            x_out = x_out.view(x_out.size(0), -1)  # [b, 2048]
            
            # Process the OTHER 3 sublayers (without gradient tracking for visualization)
            other_features = []
            for i, x_layer in enumerate(sublayers):
                if i != self.sublayer_idx:
                    with torch.no_grad():
                        x_temp = self.aide_model.model_min.conv1(x_layer)
                        x_temp = self.aide_model.model_min.bn1(x_temp)
                        x_temp = self.aide_model.model_min.relu(x_temp)
                        x_temp = self.aide_model.model_min.maxpool(x_temp)
                        x_temp = self.aide_model.model_min.layer1(x_temp)
                        x_temp = self.aide_model.model_min.layer2(x_temp)
                        x_temp = self.aide_model.model_min.layer3(x_temp)
                        x_temp = self.aide_model.model_min.layer4(x_temp)
                        x_temp = self.aide_model.model_min.avgpool(x_temp)
                        x_temp = x_temp.view(x_temp.size(0), -1)
                        other_features.append(x_temp)
            
            # Average all 4 patchwise features (1 with gradients + 3 without)
            x_patchwise = x_out  # Start with the target (has gradients)
            for feat in other_features:
                x_patchwise = x_patchwise + feat
            x_patchwise = x_patchwise / 4  # [b, 2048]
            
            # Process semantic branch (without gradients, as we're visualizing patchwise)
            with torch.no_grad():
                clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
                clip_mean = clip_mean.to(tokens.device, non_blocking=True).view(3, 1, 1)
                clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
                clip_std = clip_std.to(tokens.device, non_blocking=True).view(3, 1, 1)
                dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens.device, non_blocking=True).view(3, 1, 1)
                dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens.device, non_blocking=True).view(3, 1, 1)
                
                normalized_tokens = tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
                local_convnext_image_feats = self.aide_model.openclip_convnext_xxl(normalized_tokens)
                local_convnext_image_feats = self.aide_model.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
                x_semantic = self.aide_model.convnext_proj(local_convnext_image_feats)
            
            # Zero out semantic (as per original AIDE forward)
            # x_semantic = x_semantic * 0
            
            # Combine and classify
            x_combined = torch.cat([x_semantic, x_patchwise], dim=1)  # [b, 256 + 2048]
            output = self.aide_model.fc(x_combined)  # [b, 2]
            
            return output
            
        elif self.branch == 'semantic':
            # Extract all sublayers
            x_minmin = x[:, 0]
            x_maxmax = x[:, 1]
            x_minmin1 = x[:, 2]
            x_maxmax1 = x[:, 3]
            tokens = x[:, 4]  # [b, c, h, w]
            
            # Process semantic branch WITH gradients (this is what we're visualizing)
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens.device, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens.device, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens.device, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens.device, non_blocking=True).view(3, 1, 1)
            
            normalized_tokens = tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            
            # Make sure this requires gradients for CAM
            normalized_tokens.requires_grad_(True)
            
            # Forward through ConvNeXt (where CAM will hook!)
            local_convnext_image_feats = self.aide_model.openclip_convnext_xxl(normalized_tokens)
            local_convnext_image_feats = self.aide_model.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            x_semantic = self.aide_model.convnext_proj(local_convnext_image_feats)  # [b, 256]
            
            # Process patchwise branch WITHOUT gradients (as we're visualizing semantic)
            with torch.no_grad():
                x_minmin = self.aide_model.hpf(x_minmin)
                x_maxmax = self.aide_model.hpf(x_maxmax)
                x_minmin1 = self.aide_model.hpf(x_minmin1)
                x_maxmax1 = self.aide_model.hpf(x_maxmax1)
                
                x_min = self.aide_model.model_min(x_minmin)
                x_max = self.aide_model.model_max(x_maxmax)
                x_min1 = self.aide_model.model_min(x_minmin1)
                x_max1 = self.aide_model.model_max(x_maxmax1)
                
                x_patchwise = (x_min + x_max + x_min1 + x_max1) / 4  # [b, 2048]
            
            # Zero out semantic (as per original AIDE forward)
            x_semantic = x_semantic * 0
            
            # Combine and classify
            x_combined = torch.cat([x_semantic, x_patchwise], dim=1)  # [b, 256 + 2048]
            output = self.aide_model.fc(x_combined)  # [b, 2]
            
            return output
        
        else:
            raise ValueError(f"Unknown branch: {self.branch}")


# ============================================================================
# AIDE MODEL WITH GRADCAM++ SUPPORT
# ============================================================================
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
        
        self.fc = Mlp(2048 + 256 , 1024, 2)

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

    # ========================================================================
    # GRADCAM++ METHODS
    # ========================================================================
    def get_target_layers(self, branch='patchwise'):
        """
        Returns target layers for Grad-CAM++ visualization
        
        Args:
            branch: 'patchwise' for ResNet features or 'semantic' for ConvNeXt features
            
        Returns:
            List of target layers for Grad-CAM++
        """
        if branch == 'patchwise':
            return [self.model_min.layer4[-1]]
        elif branch == 'semantic':
            return [self.openclip_convnext_xxl.stages[-1].blocks[-1]]
        else:
            raise ValueError(f"Unknown branch: {branch}. Use 'patchwise' or 'semantic'")
    
    def create_gradcam_wrapper(self, branch='patchwise', sublayer_idx=0):
        """
        Creates a wrapper for Grad-CAM++ visualization
        
        Args:
            branch: 'patchwise' or 'semantic'
            sublayer_idx: For patchwise - which sublayer to visualize
                         0=minmin, 1=maxmax, 2=minmin1, 3=maxmax1
                         
        Returns:
            AIDEGradCAMWrapper instance
        """
        return AIDEGradCAMWrapper(self, branch=branch, sublayer_idx=sublayer_idx)

    def forward(self, x):

        b, t, c, h, w = x.shape

        x_minmin = x[:, 0] #[b, c, h, w]
        x_maxmax = x[:, 1]
        x_minmin1 = x[:, 2]
        x_maxmax1 = x[:, 3]
        tokens = x[:, 4]

        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        with torch.no_grad():
            
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)

            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            ) #[b, 3072, 8, 8]
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
            x_0 = self.convnext_proj(local_convnext_image_feats)

        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)

        x_1 = (x_min + x_max + x_min1 + x_max1) / 4
        x_0 *= 0

        x = torch.cat([x_0, x_1], dim=1)

        x = self.fc(x)

        return x


# ============================================================================
# GRADCAM ANALYZER CLASS - INTEGRATED INTO THE SAME FILE
# ============================================================================
class GradCAMAnalyzer:
    """
    Complete Grad-CAM++ analyzer for AIDE model
    This can be used with the main training/evaluation code
    """
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: AIDE_Model instance (already loaded with weights)
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.device = device
        self.model.eval()


    def generate_gradcam(self, input_tensor, target_class=None, branch='patchwise', sublayer_idx=0):
        """
        Generate Grad-CAM++ heatmap
        """
        try:
            from pytorch_grad_cam import GradCAMPlusPlus
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError:
            raise ImportError("Please install pytorch-grad-cam: pip install grad-cam")
        
        # CRITICAL: Ensure input requires gradients
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Create wrapper
        wrapper = self.model.create_gradcam_wrapper(branch=branch, sublayer_idx=sublayer_idx)
        target_layers = wrapper.get_target_layers()
        wrapper.eval()  # Eval mode
        # But enable gradients for the input
        input_tensor.requires_grad_(True)
        # Initialize Grad-CAM++ with reshape_transform for proper handling
        if branch == 'semantic':
            # ConvNeXt outputs might need reshaping
            cam = GradCAMPlusPlus(
                model=wrapper, 
                target_layers=target_layers
            )
        else:
            cam = GradCAMPlusPlus(
                model=wrapper, 
                target_layers=target_layers
            )
        
        # Get prediction
        with torch.no_grad():
            output = wrapper(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        print(f"Prediction: {pred_class}, Confidence: {confidence:.4f}")
        
        # Generate CAM - use predicted class if not specified
        if target_class is None:
            target_class = pred_class
            
        targets = [ClassifierOutputTarget(target_class)]
        
        # CRITICAL: Pass input tensor that requires grad
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)
        
        print(f"CAM stats - Min: {grayscale_cam.min():.4f}, Max: {grayscale_cam.max():.4f}, Mean: {grayscale_cam.mean():.4f}")
        
        return grayscale_cam[0], pred_class, confidence
    # def generate_gradcam(self, input_tensor, target_class=None, branch='patchwise', sublayer_idx=0):
    #     """
    #     Generate Grad-CAM++ heatmap
        
    #     Args:
    #         input_tensor: Input tensor [1, 5, 3, H, W]
    #         target_class: Target class for visualization (0 or 1)
    #         branch: 'patchwise' or 'semantic'
    #         sublayer_idx: Which sublayer to visualize (0-3)
            
    #     Returns:
    #         grayscale_cam: Heatmap
    #         pred_class: Predicted class
    #         confidence: Prediction confidence
    #     """
    #     try:
    #         from pytorch_grad_cam import GradCAMPlusPlus
    #         from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    #     except ImportError:
    #         raise ImportError("Please install pytorch-grad-cam: pip install grad-cam")
        
    #     # Create wrapper
    #     wrapper = self.model.create_gradcam_wrapper(branch=branch, sublayer_idx=sublayer_idx)
    #     target_layers = wrapper.get_target_layers()
        
    #     # Initialize Grad-CAM++
    #     cam = GradCAMPlusPlus(model=wrapper, target_layers=target_layers)
        
    #     # Get prediction
    #     with torch.no_grad():
    #         output = self.model(input_tensor)
    #         probs = F.softmax(output, dim=1)
    #         pred_class = torch.argmax(probs, dim=1).item()
    #         confidence = probs[0, pred_class].item()
        
    #     # Generate CAM
    #     if target_class is None:
    #         target_class = pred_class
            
    #     targets = [ClassifierOutputTarget(target_class)]
    #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
    #     return grayscale_cam[0], pred_class, confidence
    
    def visualize_all_branches(self, input_tensor, output_dir, image_name, true_label=None):
        """
        Generate Grad-CAM++ for all branches and sublayers
        
        Args:
            input_tensor: Input tensor [1, 5, 3, H, W]
            output_dir: Directory to save visualizations
            image_name: Name for saving files
            true_label: Ground truth label (optional)
        """
        try:
            from pytorch_grad_cam.utils.image import show_cam_on_image
        except ImportError:
            raise ImportError("Please install pytorch-grad-cam: pip install grad-cam")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get original image for visualization (use the first sublayer)
        original_img = input_tensor[0, 0].cpu().permute(1, 2, 0).numpy()
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        original_img = original_img.astype(np.float32)
        
        results = {}
        
        # Visualize patchwise branch (all 4 sublayers)
        sublayer_names = ['minmin', 'maxmax', 'minmin1', 'maxmax1']
        for idx, name in enumerate(sublayer_names):
            try:
                cam, pred, conf = self.generate_gradcam(
                    input_tensor, branch='patchwise', sublayer_idx=idx
                )
                # DEBUG: Print CAM shape
                print(f"CAM shape for {name}: {cam.shape}")
                
                # Ensure CAM is 2D
                if len(cam.shape) == 3:
                    cam = cam[:, :, 0]  # Take first channel if 3D
                elif len(cam.shape) == 1:
                    # Reshape if flattened
                    h, w = original_img.shape[:2]
                    cam = cam.reshape(h, w)
                # Create visualization
                visualization = show_cam_on_image(original_img, cam, use_rgb=True)
                
                # Save
                output_path = os.path.join(output_dir, f'{image_name}_patchwise_{name}.png')
                Image.fromarray(visualization).save(output_path)
                
                results[f'patchwise_{name}'] = {
                    'cam': cam,
                    'pred': pred,
                    'confidence': conf,
                    'path': output_path
                }
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error generating CAM for {name}: {e}")
        
        # Visualize semantic branch
        try:
            cam, pred, conf = self.generate_gradcam(
                input_tensor, branch='semantic'
            )
            # DEBUG: Print CAM shape
            print(f"CAM shape for semantic: {cam.shape}")
            
            # Ensure CAM is 2D
            if len(cam.shape) == 3:
                cam = cam[:, :, 0]
            elif len(cam.shape) == 1:
                h, w = original_img.shape[:2]
                cam = cam.reshape(h, w)
              
            visualization = show_cam_on_image(original_img, cam, use_rgb=True)
            output_path = os.path.join(output_dir, f'{image_name}_semantic.png')
            Image.fromarray(visualization).save(output_path)
            
            results['semantic'] = {
                'cam': cam,
                'pred': pred,
                'confidence': conf,
                'path': output_path
            }
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error generating semantic CAM: {e}")
        
        # Save original image
        original_save_path = os.path.join(output_dir, f'{image_name}_original.png')
        Image.fromarray((original_img * 255).astype(np.uint8)).save(original_save_path)
        
        return results
    
    def analyze_dataset(self, data_loader, output_root, max_samples=100, save_fp_fn=True):
        """
        Analyze dataset and generate Grad-CAMs for interesting cases
        
        Args:
            data_loader: DataLoader for the dataset
            output_root: Root directory for saving results
            max_samples: Maximum number of samples to analyze
            save_fp_fn: Whether to save False Positives and False Negatives
        """
        os.makedirs(output_root, exist_ok=True)
        
        fp_samples = []  # False positives
        fn_samples = []  # False negatives
        tp_samples = []  # True positives
        tn_samples = []  # True negatives
        
        print("Analyzing dataset...")
        for idx, batch in enumerate(tqdm(data_loader)):
            if idx >= max_samples:
                break
            
            if len(batch) == 2:
                images, labels = batch
            else:
                images = batch[0]
                labels = batch[1] if len(batch) > 1 else None
                
            images = images.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
            
            if labels is not None:
                for i in range(len(images)):
                    pred = preds[i].item()
                    true_label = labels[i].item()
                    
                    # Categorize
                    if pred == 1 and true_label == 0:
                        fp_samples.append((idx, i, images[i:i+1], true_label))
                    elif pred == 0 and true_label == 1:
                        fn_samples.append((idx, i, images[i:i+1], true_label))
                    elif pred == 1 and true_label == 1:
                        tp_samples.append((idx, i, images[i:i+1], true_label))
                    else:
                        tn_samples.append((idx, i, images[i:i+1], true_label))
        
        print(f"\nDataset Statistics:")
        print(f"False Positives: {len(fp_samples)}")
        print(f"False Negatives: {len(fn_samples)}")
        print(f"True Positives: {len(tp_samples)}")
        print(f"True Negatives: {len(tn_samples)}")
        
        if save_fp_fn:
            # Generate Grad-CAMs for False Positives
            print("\nGenerating Grad-CAMs for False Positives...")
            fp_dir = os.path.join(output_root, 'false_positives')
            for idx, (batch_idx, img_idx, img_tensor, true_label) in enumerate(fp_samples[:20]):
                self.visualize_all_branches(
                    img_tensor, 
                    fp_dir, 
                    f'FP_{idx}_batch{batch_idx}_img{img_idx}',
                    true_label=true_label
                )
            
            # Generate Grad-CAMs for False Negatives
            print("\nGenerating Grad-CAMs for False Negatives...")
            fn_dir = os.path.join(output_root, 'false_negatives')
            for idx, (batch_idx, img_idx, img_tensor, true_label) in enumerate(fn_samples[:20]):
                self.visualize_all_branches(
                    img_tensor, 
                    fn_dir, 
                    f'FN_{idx}_batch{batch_idx}_img{img_idx}',
                    true_label=true_label
                )
        
        # Save statistics
        stats = {
            'fp': len(fp_samples),
            'fn': len(fn_samples),
            'tp': len(tp_samples),
            'tn': len(tn_samples),
            'total': len(fp_samples) + len(fn_samples) + len(tp_samples) + len(tn_samples),
            'accuracy': (len(tp_samples) + len(tn_samples)) / (len(fp_samples) + len(fn_samples) + len(tp_samples) + len(tn_samples))
        }
        
        import json
        stats_path = os.path.join(output_root, 'analysis_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"\nStatistics saved to: {stats_path}")
        
        return stats


def AIDE(resnet_path, convnext_path):
    model = AIDE_Model(resnet_path, convnext_path)
    return model




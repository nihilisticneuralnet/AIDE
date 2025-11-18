import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import clip
import open_clip
from .srm_filter_kernel import all_normalized_hpf_list
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn.functional as F

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

    

    def forward(self, x, return_cam=False, cam_branches=None):

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
        # x_1 = (x_min + x_max + x_min1 + x_max1) / 40
        # x_1 *= 9
        x_0 *= 0

        x = torch.cat([x_0, x_1], dim=1)
        
        # my code
        # x_0 = semantic, x_1 = patchwise
        # x = torch.cat([0.1 * x_0, 0.9 * x_1], dim=1)

        x = self.fc(x)
      
        if return_cam:
            if cam_branches is None:
                cam_branches = ['pfe_high', 'pfe_low', 'sfe']
            # Note: CAM generation is done separately in evaluate() to avoid autograd issues
            return x

        return x

    def get_target_layers(self, branch='pfe_high'):
        """
        Returns target layers for Grad-CAM++
        branch: 'pfe_high', 'pfe_low', or 'sfe'
        """
        if branch == 'pfe_high':
            # Target the final conv3 in layer4 of model_max (highest frequency)
            return [self.model_max.layer4[-1].conv3]
        elif branch == 'pfe_low':
            # Target the final conv3 in layer4 of model_min (lowest frequency)
            return [self.model_min.layer4[-1].conv3]
        elif branch == 'sfe':
            # Target the last conv block before projection in ConvNeXt
            # ConvNeXt stages are in self.openclip_convnext_xxl.stages
            return [self.openclip_convnext_xxl.stages[-1].blocks[-1]]
        else:
            raise ValueError(f"Unknown branch: {branch}")

    def generate_gradcam(self, input_tensor, target_class=None, branches=['pfe_high', 'pfe_low', 'sfe']):
        """
        Generate Grad-CAM++ heatmaps using manual implementation
        """
        self.train()  # Need gradients
        cams = {}
        
        # Get prediction if target_class not specified
        if target_class is None:
            with torch.no_grad():
                logits = self.forward(input_tensor)
                target_class = logits.argmax(dim=1).item()
        
        for branch in branches:
            try:
                # Storage for activations and gradients
                activations = []
                gradients = []
                
                # Hook functions
                def forward_hook(module, input, output):
                    activations.append(output.detach())
                
                def backward_hook(module, grad_input, grad_output):
                    gradients.append(grad_output[0].detach())
                
                # Register hooks on target layer
                target_layers = self.get_target_layers(branch)
                forward_handle = target_layers[0].register_forward_hook(forward_hook)
                backward_handle = target_layers[0].register_full_backward_hook(backward_hook)
                
                # Forward pass
                if branch == 'pfe_high':
                    logits = self._forward_pfe_only(input_tensor, 'pfe_high')
                elif branch == 'pfe_low':
                    logits = self._forward_pfe_only(input_tensor, 'pfe_low')
                elif branch == 'sfe':
                    logits = self._forward_sfe_only(input_tensor)
                
                # Backward pass
                self.zero_grad()
                score = logits[:, target_class]
                score.backward()
                
                # Remove hooks
                forward_handle.remove()
                backward_handle.remove()
                
                # Compute Grad-CAM++
                feature_map = activations[0]  # [1, C, H, W]
                grads = gradients[0]  # [1, C, H, W]
                
                # Grad-CAM++ weights
                alpha = grads.pow(2)
                alpha = alpha / (2 * alpha + (feature_map * grads.pow(3)).sum(dim=(2, 3), keepdim=True) + 1e-8)
                
                weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
                
                # Generate CAM
                cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))  # [1, 1, H, W]
                
                # Normalize
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                
                # Resize to input size
                h, w = input_tensor.shape[-2:]
                cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
                
                # Convert to numpy
                cams[branch] = cam[0, 0].cpu().numpy()
                
            except Exception as e:
                print(f"Error generating CAM for branch {branch}: {e}")
                import traceback
                traceback.print_exc()
                h, w = input_tensor.shape[-2:]
                cams[branch] = np.zeros((h, w))
        
        self.eval()  # Back to eval mode
        return cams

    def _forward_pfe_only(self, x, branch):
        """Forward pass through PFE branch only - ensure gradients enabled"""
        b, t, c, h, w = x.shape
        
        if branch == 'pfe_high':
            x_branch = x[:, 1]  # maxmax
            x_hpf = self.hpf(x_branch)
            features = self.model_max(x_hpf)
        else:  # pfe_low
            x_branch = x[:, 0]  # minmin
            x_hpf = self.hpf(x_branch)
            features = self.model_min(x_hpf)
        
        # Return logits (don't use torch.no_grad here)
        return self.fc(torch.cat([torch.zeros(b, 256, device=x.device), features], dim=1))
    
    def _forward_sfe_only(self, x):
        """Forward pass through SFE branch only"""
        b, t, c, h, w = x.shape
        tokens = x[:, 4]
        
        # Note: ConvNeXt is frozen, but we still need gradients from its output
        clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(tokens).view(3, 1, 1)
        clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(tokens).view(3, 1, 1)
        dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens).view(3, 1, 1)
        dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens).view(3, 1, 1)
        
        local_convnext_image_feats = self.openclip_convnext_xxl(
            tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
        )
        local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(b, -1)
        features = self.convnext_proj(local_convnext_image_feats)
        
        # Return logits
        return self.fc(torch.cat([features, torch.zeros(b, 2048, device=x.device)], dim=1))

def AIDE(resnet_path, convnext_path):
    model = AIDE_Model(resnet_path, convnext_path)
    return model





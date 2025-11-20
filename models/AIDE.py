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

    

    def forward(self, x, return_cam=False, cam_branches=None, branch_mask=None):
        """
        branch_mask: dict or None, e.g. {'pfe_high':1.0, 'pfe_low':0.0, 'sfe':0.0}
        return_cam: same semantics as before (keeps behavior)
        Returns logits (and same as before). This forward also supports masking branches
        while preserving the original computation graph.
        """
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
    
        # semantic branch preprocessing (ConvNeXt) - keep graph intact
        clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(tokens, non_blocking=True).view(3, 1, 1)
        clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(tokens, non_blocking=True).view(3, 1, 1)
        dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(tokens, non_blocking=True).view(3, 1, 1)
        dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(tokens, non_blocking=True).view(3, 1, 1)
    
        local_convnext_image_feats = self.openclip_convnext_xxl(
            tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
        )  #[b, 3072, 8, 8]
        assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
        local_convnext_image_feats_pooled = self.avgpool(local_convnext_image_feats).view(tokens.size(0), -1)
        x_0 = self.convnext_proj(local_convnext_image_feats_pooled)  # [b, 256]
    
        # PFE branches (ResNets)
        x_min = self.model_min(x_minmin)   # [b, 2048]
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)
    
        x_1 = (x_min + x_max + x_min1 + x_max1) / 4  # [b, 2048]
    
        # default masks: keep both branches as in training/inference
        if branch_mask is None:
            mask_pfe = 1.0
            mask_sfe = 1.0
        else:
            mask_pfe = float(branch_mask.get('pfe', branch_mask.get('pfe_high', 1.0)))
            mask_sfe = float(branch_mask.get('sfe', 1.0))
    
        # preserve original scale behavior; allow masking for CAM
        x = torch.cat([mask_sfe * x_0, mask_pfe * x_1], dim=1)  # [b, 256+2048]
    
        x = self.fc(x)
    
        if return_cam:
            if cam_branches is None:
                cam_branches = ['pfe_high', 'pfe_low', 'sfe']
            # CAM generation done externally; just return logits here
            return x
    
        return x


    def get_target_layers(self, branch='pfe_high'):
        """
        Return module(s) on which to register forward hooks that capture spatial activations.
        For ResNet PFE branches we hook the output of the last conv block (before pooling).
        For SFE we hook the ConvNeXt trunk output before avgpool (which is [B, 3072, 8, 8]).
        """
        if branch == 'pfe_high':
            # hook the last block module (the whole Bottleneck block) so forward hook captures spatial map
            return [self.model_max.layer4[-1]]
        elif branch == 'pfe_low':
            return [self.model_min.layer4[-1]]
        elif branch == 'sfe':
            # hook the convnext trunk output stage (the module that produces the spatial tensor)
            # openclip_convnext_xxl is the visual.trunk.visual trunk — stages[-1] contains the final blocks
            return [self.openclip_convnext_xxl.stages[-1]]
        else:
            raise ValueError(f"Unknown branch: {branch}")


    def generate_gradcam(self, input_tensor, target_class=None, branches=['pfe_high', 'pfe_low', 'sfe']):
        """
        Grad-CAM++ generator per-branch. Expects:
          - input_tensor on CPU or same device as model (will be moved to device)
          - self.get_target_layers(branch) returns a list where [0] is an nn.Module to hook
          - self.forward(x, branch_mask=mask) accepts branch_mask if you use it (optional)
        """
        self.eval()
    
        # --- unify target_class into list ---
        if target_class is None:
            with torch.no_grad():
                logits = self.forward(input_tensor.to(next(self.parameters()).device))
                target_class = logits.argmax(dim=1).item()
    
        if isinstance(target_class, int):
            target_classes = [target_class]
        elif isinstance(target_class, torch.Tensor):
            target_classes = target_class.detach().cpu().tolist()
        elif isinstance(target_class, list):
            target_classes = target_class
        else:
            raise ValueError(f"Invalid target_class type: {type(target_class)}")
    
        device = next(self.parameters()).device
        cams = {}
    
        # mapping branch -> slice index within the T-frame input (adjust if your temporal dim differs)
        branch_to_index = {'pfe_low': 0, 'pfe_high': 1, 'sfe': 4}
    
        def _set_convnext_requires_grad(flag):
            if hasattr(self, "openclip_convnext_xxl") and self.openclip_convnext_xxl is not None:
                for p in self.openclip_convnext_xxl.parameters():
                    p.requires_grad = flag
    
        for branch in branches:
            if branch not in branch_to_index:
                cams[branch] = None
                continue
    
            target_idx = branch_to_index[branch]
            activations = []
    
            # branch isolation mask for forward - used in your forward optionally
            if 'pfe' in branch:
                mask = {'pfe': 1.0, 'sfe': 0.0}
            elif branch == 'sfe':
                mask = {'pfe': 0.0, 'sfe': 1.0}
            else:
                raise ValueError(f"Unknown branch: {branch}")
    
            def forward_hook(module, input, output):
                # capture activation tensor (module output)
                activations.append(output)
    
            # get hook module
            target_layers = self.get_target_layers(branch)
            if not isinstance(target_layers, (list, tuple)) or len(target_layers) == 0:
                raise RuntimeError(f"get_target_layers returned unexpected value for branch {branch}: {target_layers}")
            hook_module = target_layers[0]
            handle = hook_module.register_forward_hook(forward_hook)
    
            try:
                # Prepare input preserving computational graph
                # Assumes input_tensor shape is [B, T, C, H, W] (if different, adjust accordingly)
                x = input_tensor.clone().to(device)
                # make x require grad so gradients flow to intermediate activations
                x.requires_grad_(True)
    
                # Build a mask that keeps only target slice and zeros-out others while preserving graph
                # This avoids detach/replace which breaks autograd path
                # Create a mask of same shape as x with zeros except 1 at target index
                # We try to handle both [B, T, C, H, W] and [B, C, H, W] cases:
                if x.dim() == 5:
                    B, T, C, H, W = x.shape
                    if target_idx < 0 or target_idx >= T:
                        raise IndexError(f"target_idx {target_idx} out of range for temporal length {T}")
                    mask_tensor = torch.zeros_like(x, device=device)
                    mask_tensor[:, target_idx] = 1.0
                elif x.dim() == 4:
                    # if no temporal dim, treat target_idx as channel-grouping index not supported here
                    # fallback: use full input (no masking)
                    mask_tensor = torch.ones_like(x, device=device)
                else:
                    raise RuntimeError(f"Unsupported input tensor dimensionality: {x.dim()}")
    
                # masked input preserves graph (keeps other frames zeroed but connected)
                x_masked = x * mask_tensor
    
                convnext_was_frozen = False
                if branch == 'sfe':
                    # temporarily enable convnext grads if they were frozen
                    _set_convnext_requires_grad(True)
                    convnext_was_frozen = True
    
                with torch.enable_grad():
                    logits = self.forward(x_masked, branch_mask=mask) if 'branch_mask' in self.forward.__code__.co_varnames else self.forward(x_masked)
                    if logits is None:
                        raise RuntimeError("Model forward returned None logits during CAM generation")
    
                    # safer aggregation: sum the scores for chosen target class across batch (works any batch size)
                    # pick the first target class in target_classes list
                    cls = target_classes[0]
                    # ensure logits shape is [B, num_classes]
                    if logits.dim() == 1:
                        # single-class logit vector (rare), convert to shape [1]
                        score = logits if logits.shape[0] == 1 else logits.sum()
                    else:
                        score = logits[:, cls].sum()
    
                    # ensure activation captured
                    if len(activations) == 0:
                        raise RuntimeError(f"No activations captured for branch {branch}; check hook target layer")
    
                    act = activations[0]  # expected shape [B, C, H, W]
                    # retain grad on activation to inspect grads after backward
                    act.retain_grad()
    
                    # zero grads
                    self.zero_grad()
                    # backprop the scalar score
                    score.backward(retain_graph=True)
    
                    # check grad presence
                    if act.grad is None:
                        raise RuntimeError(f"Activation gradient is None for branch {branch}; autograd graph likely broken")
    
                    grads = act.grad  # [B, C, H, W]
                    feature_map = act.detach()
    
                    # Grad-CAM++ computations (your existing formula)
                    grads_power_2 = grads.pow(2)
                    grads_power_3 = grads.pow(3)
                    spatial_sum = (feature_map * grads_power_3).sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
                    denom = 2 * grads_power_2 + spatial_sum + 1e-8
                    alpha = grads_power_2 / denom  # [B, C, H, W]
                    weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
                    cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))  # [B,1,H,W]
    
                    # normalize per-sample
                    cam_min = cam.view(cam.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
                    cam = cam - cam_min
                    cam_max = cam.view(cam.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
                    cam = cam / (cam_max + 1e-8)
    
                    # upsample to input image HxW
                    if x.dim() == 5:
                        H_in, W_in = x.shape[-2], x.shape[-1]
                    else:
                        H_in, W_in = input_tensor.shape[-2], input_tensor.shape[-1]
    
                    cam_up = F.interpolate(cam, size=(H_in, W_in), mode='bilinear', align_corners=False)
                    cams[branch] = cam_up[0, 0].detach().cpu().numpy()
    
                # restore convnext freeze state
                if convnext_was_frozen:
                    _set_convnext_requires_grad(False)
    
            except Exception as e:
                h, w = input_tensor.shape[-2], input_tensor.shape[-1]
                print(f"[generate_gradcam] Error for branch {branch}: {e}")
                cams[branch] = np.zeros((h, w))
            finally:
                handle.remove()
    
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














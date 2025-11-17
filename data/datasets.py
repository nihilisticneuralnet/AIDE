# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from PIL import Image
import io
import torch
from .dct import DCT_base_Rec_Module
import random


def pil_to_padded_tensor(pil_img, multiple=16):
    """
    Convert PIL image to a tensor (C,H,W) float in [0,1], and pad right/bottom
    so H and W are divisible by `multiple`. Returns a 4-D tensor (1,C,Hp,Wp)
    which is what Kornia's augmentations expect (batch dimension).
    """
    # to_tensor returns float in range [0,1] and shape C,H,W
    t = TF.to_tensor(pil_img).float()
    _, h, w = t.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return t.unsqueeze(0)   # add batch dim
    # F.pad expects pad = (left, right, top, bottom)
    padded = F.pad(t, (0, pad_w, 0, pad_h))
    return padded.unsqueeze(0)


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import kornia.augmentation as K

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)

transform_before = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0])
    # transforms.Lambda(lambda pil_img: Perturbations(pil_to_padded_tensor(pil_img))[0])
    ]
)
transform_before_test = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform_train = transforms.Compose([
        transforms.Lambda(
        lambda img: TF.center_crop(
            img, 
            min(img.shape[1], img.shape[2])  # crop largest square from center
        )
    ),
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


class TrainDataset(Dataset):
    def __init__(self, is_train, args):
        
        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []

        if'GenImage' in root and root.split('/')[-1] != 'train':
            file_path = root

            if '0_real' not in os.listdir(file_path):
                for folder_name in os.listdir(file_path):
                
                    assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                    for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                 
                    for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
            
            else:
                for image_path in os.listdir(os.path.join(file_path, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
        else:
            
            # og code

            # for filename in os.listdir(root):

            #     file_path = os.path.join(root, filename)

            #     if '0_real' not in os.listdir(file_path):
            #         for folder_name in os.listdir(file_path):
                    
            #             assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

            #             for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
            #                 self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                    
            #             for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
            #                 self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
                
            #     else:
            #         for image_path in os.listdir(os.path.join(file_path, '0_real')):
            #             self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
            #         for image_path in os.listdir(os.path.join(file_path, '1_fake')):
            #             self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
            
            label_map = {'0_real': 0, '1_fake': 1}

            # Iterate through the items in the label_map
            for folder_name, label in label_map.items():
                # Create the full path to the folder (e.g., 'path/to/root/0_real')
                folder_path = os.path.join(root, folder_name)
                
                # Check if the directory exists to prevent errors
                if not os.path.isdir(folder_path):
                    continue # Skip if the folder doesn't exist
                    
                # List all image files within that folder
                for image_filename in os.listdir(folder_path):
                    # Create the full path to the image file
                    image_path = os.path.join(folder_path, image_filename)
                    
                    # Append the image path and its corresponding label to your data list
                    self.data_list.append({"image_path": image_path, "label": label})
            

                
        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))


        image = transform_before(image)

        try:
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
        except:
            print(f'image error: {image_path}, c, h, w: {image.shape}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        


        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))

    

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        
        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []

        file_path = root

        # if '0_real' not in os.listdir(file_path):
        #     for folder_name in os.listdir(file_path):
    
        #         assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']
                
        #         for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
        #             self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                
        #         for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
        #             self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
        
        # else:
        #     for image_path in os.listdir(os.path.join(file_path, '0_real')):
        #         self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
        #     for image_path in os.listdir(os.path.join(file_path, '1_fake')):
        #         self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
        
        # label_map = {'0_real': 0, '1_fake': 1}

        # for folder_name, label in label_map.items():
        #     folder_path = os.path.join(root, folder_name)

        #     # It's good practice to check if the folder actually exists.
        #     if not os.path.isdir(folder_path):
        #         print(f"Warning: Directory not found, skipping: {folder_path}")
        #         continue

        #     # Loop through each image in the folder.
        #     for image_filename in os.listdir(folder_path):
        #         # Create the full, platform-independent path to the image.
        #         image_path = os.path.join(folder_path, image_filename)

        #         # Add the image path and its label to your list.
        #         self.data_list.append({"image_path": image_path, "label": label})
        
        folder_name = os.path.basename(root)  # This will be '0_real' or '1_fake'

        # 2. Determine the label from the folder name
        if folder_name == '0_real':
            label = 0
        elif folder_name == '1_fake':
            label = 1
        else:
            # This handles other datasets like 'progan', 'stylegan', etc.
            # You must decide what label they get.
            # If you are *only* testing '0_real' and '1_fake', you can be stricter.
            print(f"Warning: Unrecognized folder name '{folder_name}'. Assuming label 1 (fake).")
            label = 1 # Default to fake if not '0_real'

        # 3. Check if the given path is valid
        if not os.path.isdir(root):
            print(f"Error: Directory not found or is not a directory: {root}")
            return  # Stop initialization

        # 4. Load all images *directly* from that folder
        for image_filename in os.listdir(root):
            image_path = os.path.join(root, image_filename)
            
            # Make sure it's a file
            if os.path.isfile(image_path):
                self.data_list.append({"image_path": image_path, "label": label})
        


        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        image = Image.open(image_path).convert('RGB')

        image = transform_before_test(image)

        # x_max, x_min, x_max_min, x_minmin = self.dct(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)


        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        
        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))



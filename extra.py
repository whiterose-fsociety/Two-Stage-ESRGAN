# Copy Parts that make sense
import torchvision.transforms.functional as T
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
from glob import glob,iglob
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2


dir_name="train"
hr_image_dir =  "DIV2K_{}_HR".format(dir_name)
lr_image_dir = "DIV2K_{}_LR_x8_unknown".format(dir_name)
hr_root_dir = "../Datasets/{}".format(hr_image_dir)
lr_root_dir = "../Datasets/{}".format(lr_image_dir)
hr_data = sorted(glob(os.path.join(hr_root_dir,"*.png")))
lr_data = sorted(glob(os.path.join(lr_root_dir,"*.png")))
lr_data[0],hr_data[0]
l = cv2.imread(lr_data[0])
h = cv2.imread(hr_data[0])

image_transforms = A.Compose([A.Normalize(mean=[0,0,0],std=[1,1,1]),ToTensorV2()])
hr = image_transforms(image=h)['image'] # Get Tensor version of hr image
lr = image_transforms(image=l)['image'] # Get Tensor version of lr image

i, j, h, w = torchvision.transforms.RandomCrop.get_params(hr,output_size=(128,128)) # Get random crop of tensor hr image


hr_pil = torchvision.transforms.ToPILImage()(hr) # Pil Version of hr_image
lr_pil = torchvision.transforms.ToPILImage()(lr) # Pil Version of lr_image
hcrop = TF.crop(hr_pil,i,j,h,w) # Get 128x128 crop of hr image
lcrop = TF.crop(lr_pil,i//4,j//4,h//4,w//4) # Get 128x128 crop of lr image


to_tensor = A.Compose([ToTensorV2()]) 

hcrop_tensor = to_tensor(image=np.asarray(hcrop))['image'] # Tensor version of 128 crop
lcrop_tensor = to_tensor(image=np.asarray(lcrop))['image'] # Tensor version of 128 crop


hcrop_cv = hcrop_tensor.cpu().detach().permute(1,2,0)
lcrop_cv = lcrop_tensor.cpu().detach().permute(1,2,0)

plt.imshow(hcrop_cv)
plt.savefig("hcrop.png")
plt.imshow(lcrop_cv)
plt.savefig("lcrop.png")

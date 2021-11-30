import torchvision.transforms.functional as TF
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

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
# image_transforms = A.Compose([A.Normalize(mean=[0,0,0],std=[1,1,1]),ToTensorV2()]) # Weights 1
# image_transforms = A.Compose([A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),ToTensorV2()]) # Weights 2
image_transforms = A.Compose([A.Normalize(mean=MEAN,std=STD),ToTensorV2()]) # Weights 3 / 8


# to_tensor = A.Compose([A.Normalize(mean=[0,0,0],std=[1,1,1]),ToTensorV2()]) # Weights 1
# to_tensor = A.Compose([A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),ToTensorV2()]) # Weights 2
to_tensor = A.Compose([A.Normalize(mean=MEAN,std=STD),ToTensorV2()]) # Weights 3 / 8

class ImageFolder(Dataset):
    def __init__(self,dir_name="train",for_hr=False):
        hr_image_dir =  "DIV2K_{}_HR".format(dir_name)
        lr_image_dir = "DIV2K_{}_LR_x8_unknown".format(dir_name)
        hr_root_dir = "../Datasets/{}".format(hr_image_dir)
        lr_root_dir = "../Datasets/{}".format(lr_image_dir)
        self.hr_data = sorted(glob(os.path.join(hr_root_dir,"*.png")))
        self.lr_data = sorted(glob(os.path.join(lr_root_dir,"*.png")))
        self.for_hr = for_hr
        
    def __len__(self):
        return len(self.hr_data)
    
    def __getitem__(self,index):
        lr_image = np.asarray(Image.open(self.lr_data[index]))
        hr_image = np.asarray(Image.open(self.hr_data[index]))

        hr_image = image_transforms(image=hr_image)['image'] # Get Tensor version of hr image
        lr_image = image_transforms(image=lr_image)['image'] # Get Tensor version of lr image
        downsampled_hr_image = F.avg_pool2d(hr_image,4,4)
        
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(hr_image,output_size=(config.HIGH_RES,config.HIGH_RES)) # Get random crop of tensor hr image


        hr_pil = torchvision.transforms.ToPILImage()(hr_image) # Pil Version of hr_image
        downsampled_hr_pil = torchvision.transforms.ToPILImage()(downsampled_hr_image) # Pil Version of hr_image
        lr_pil = torchvision.transforms.ToPILImage()(lr_image) # Pil Version of lr_image
        hcrop = TF.crop(hr_pil,i,j,h,w) # Get 128x128 crop of hr image
        lcrop = TF.crop(lr_pil,i//4,j//4,h//4,w//4) # Get 128x128 crop of lr image
        downsampled_crop = TF.crop(downsampled_hr_pil,i//4,j//4,h//4,w//4) # Get 128x128 crop of downsampled_hr_pil image
        

        hcrop_tensor = to_tensor(image=np.asarray(hcrop))['image'] # Tensor version of 128 crop
        lcrop_tensor = to_tensor(image=np.asarray(lcrop))['image'] # Tensor version of 128 crop
        downsampled_tensor = to_tensor(image=np.asarray(downsampled_crop))['image'] # Tensor version of 128 crop
        
        
        
        return lcrop_tensor.float(),hcrop_tensor.float(),downsampled_tensor.float()
            
    
def test():
    dataset = ImageFolder(dir_name="train")
    loader = DataLoader(dataset,batch_size=1,num_workers=0)
    for low_res,high_res,downsampled_hr_image in loader:
        print("LR Image",low_res.shape)
        print("HR Image",high_res.shape)
        print("Downsampled Image",downsampled_hr_image.shape)
        print("Sample Values: ",low_res[0,:,:3,:3])
        print("=============")
        break


test()


import torch
import utils
from PIL import Image
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples,test_models
from model import Generator, Discriminator
from dataset import ImageFolder
import numpy as np
import cv2
from psnr import *
from datasets import load_dataset
from super_image import EdsrModel
from super_image.data import EvalDataset, EvalMetrics
import matplotlib.pyplot as plt
import pandas as pd

gen = Generator(in_channels=3).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
g_optimizer = optim.Adam(gen.parameters(),     0.0001, (0.9, 0.999))  # Generator learning rate during adversarial network training.





MODEL_NAME = "Combined Enhanced Super-Resolution Generative Adversarial Network Without Penalty"
SAVE_RESULT_FOLDER = "../Quantify/results/{}".format(MODEL_NAME)
IMAGES = "../Datasets/set5/SR_testing_datasets/{}"
BSDS = IMAGES.format("BSDS100")
Set5 = IMAGES.format("Set5")
Set14 = IMAGES.format("Set14")


set5_dataset = load_dataset('eugenesiow/Set5', 'bicubic_x4', split='validation')
set14_dataset = load_dataset('eugenesiow/Set14', 'bicubic_x4', split='validation')
bsd100_dataset = load_dataset('eugenesiow/BSD100', 'bicubic_x4', split='validation')

utils.load_checkpoint(
config.CHECKPOINT_GEN_ADV,
gen,
opt_gen,
config.LEARNING_RATE,
)
set5_eval_dataset = EvalDataset(set5_dataset)
EvalMetrics().evaluate(gen, set5_eval_dataset)

# test_models(gen,BSDS,SAVE_RESULT_FOLDER,folder_name="BSDS100")
# test_models(gen,Set5,SAVE_RESULT_FOLDER,folder_name="Set5")
# test_models(gen,Set14,SAVE_RESULT_FOLDER,folder_name="Set14")
# plot_examples(gen,epoch=None,idx=None,root_folder=config.DEFAULT_ROOT_DIR,low_res_folder=config.LR_TRAIN_IMAGES,verbose=True)
# plot_examples(gen,epoch=None,idx=None,root_folder=config.DEFAULT_ROOT_DIR,low_res_folder=config.LR_VALID_IMAGES,verbose=True)

def get_image_metrics(model,lr_image,hr_image,index=0,dataset_name="set5",verbose=False):
    gen.eval()
    scale = lambda img: ((img - img.min()) * (1/(img.max() - img.min()) * 255))
    lr_image = config.test_transform(image=np.asarray(lr_image))['image'].unsqueeze(0).to(config.DEVICE)
    if dataset_name == "bsd100":    
        hr_image = config.test_transform(image=np.asarray(hr_image))['image'].unsqueeze(0).to(config.DEVICE)[:,:,:-1,:-1]
    else:
        hr_image = config.test_transform(image=np.asarray(hr_image))['image'].unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        sr_image = model(lr_image)
        
    psnr_value = PSNR.__call__(scale(hr_image[:]),scale(sr_image))
    image_hr = hr_image[0].cpu().numpy().transpose(1,2,0)
    image_sr = sr_image[0].cpu().numpy().transpose(1,2,0)
    ssim_value = SSIM._ssim(scale(image_hr),scale(image_sr))
    if not os.path.exists(f"{dataset_name}"): #
       os.makedirs(f"{dataset_name}")         
    if verbose:
        print("-------------------->Saving Image {}<--------------------".format(index))
    save_image(sr_image*0.5+0.5,f"{dataset_name}/image_{index}.png")
    # plot_images([image_hr,image_sr],["HR","SR"])
    gen.train()
    return psnr_value,ssim_value

def get_dataset_metrics(dataset,dataset_name="bsd100",verbose=False):
    if verbose:
        print(f"================Performing Image Quality Assessment on {dataset_name}================")
    psnr_values = []
    ssim_values = []
    for idx,image_pair in enumerate(dataset):
        try:
            lr_image = Image.open(dataset[idx]['lr'])
            hr_image = Image.open(dataset[idx]['hr'])
            psnr_value,ssim_value = get_image_metrics(gen,lr_image,hr_image,index=idx,dataset_name=dataset_name,verbose=verbose)
            if verbose:
                print(f"{idx}: PSNR Value = {psnr_value}, SSIM Value = {ssim_value}")
                print("-------------------------------")
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
        except:
            continue
    avg_psnr = sum(psnr_values)/len(psnr_values)
    variance_psnr = sum([((x - avg_psnr) ** 2) for x in psnr_values]) / len(psnr_values)
    std_psnr = variance_psnr ** 0.5
    
    avg_ssim = sum(ssim_values)/len(ssim_values)
    variance_ssim = sum([((x - avg_ssim) ** 2) for x in ssim_values]) / len(ssim_values)
    std_ssim = variance_ssim ** 0.5
    
    if verbose:
        print(f"{dataset_name} has the average PSNR Value = {avg_psnr} with SSIM Value = {avg_ssim}")
        print()
    return avg_psnr,std_psnr,avg_ssim,std_ssim
        
        
set5_psnr_avg,set5_psnr_std,set5_ssim_avg,set5_ssim_ssim = get_dataset_metrics(set5_dataset,dataset_name="set5",verbose=False)
set14_psnr_avg,set14_psnr_std,set14_ssim_avg,set14_ssim_ssim = get_dataset_metrics(set14_dataset,dataset_name="set14",verbose=False)
bsd100_psnr_avg,bsd100_psnr_std,bsd100_ssim_avg,bsd100_ssim_ssim = get_dataset_metrics(bsd100_dataset,dataset_name="bsd100",verbose=False)

def save_results(*results,titles=["Average PSNR","STD PSNR","Average SSIM","STD SSIM"],dataset_name="set5",verbose=True):
    if verbose:
        print(f"Saving {dataset_name} in a csv")
    results = np.asarray(results)
    dataframe = pd.DataFrame(results).T
    dataframe = dataframe.rename(columns=dict(zip(list(range(len(titles))),titles)))
    dataframe.to_csv("{}.csv".format(dataset_name))

# save_results(set5_psnr_avg,set5_psnr_std,set5_ssim_avg,set5_ssim_ssim,dataset_name="set5")
# save_results(set14_psnr_avg,set14_psnr_std,set14_ssim_avg,set14_ssim_ssim,dataset_name="set14")
# save_results(bsd100_psnr_avg,bsd100_psnr_std,bsd100_ssim_avg,bsd100_ssim_ssim,dataset_name="bsd100")
lr_image = Image.open(set5_dataset[0]['lr'])
hr_image = Image.open(set5_dataset[0]['hr'])
psnr_value,ssim_value = get_image_metrics(gen,lr_image,hr_image,index=0,dataset_name="set5",verbose=False)
print(psnr_value,ssim_value )
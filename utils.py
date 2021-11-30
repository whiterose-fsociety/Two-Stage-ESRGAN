import torch
import torch.nn as nn
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import cv2


def gradient_penalty(critic,real,fake,device='cpu'):
    BATCH_SIZE,C,H,W = real.shape
    epsilon = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images = real*epsilon + fake *(1-epsilon) # 90% real image + 10% fake
    
    
    #Calculate Critic Score
    mixed_scores = critic(interpolated_images)
    
    gradient = torch.autograd.grad(
        inputs= interpolated_images,
        outputs= mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    
    
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1) # L2 norm
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty


def save_checkpoint(model,optimizer,filename='my_checkpoint.pth.tar'):
    print("=>Saving checkpoint")
    checkpoint = {
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict()
    }
    SAVE_DIR = config.SAVE_RESULT_FOLDER.format(config.MODEL_NAME,config.SAVE_WEIGHTS)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    filesave = f"{SAVE_DIR}/{filename}"
    torch.save(checkpoint,filesave)
    
    
    
    
def load_checkpoint(checkpoint_file,model,optimizer,lr,vgg=False,esrgan=False):
    print("=>Loading checkpoint")
    checkpoint = torch.load(checkpoint_file,map_location=config.DEVICE)
    if vgg:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif esrgan:
        checkpoint = torch.load(checkpoint_file,map_location=config.DEVICE)
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint)
        optimizer.load_state_dict(checkpoint)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
            
        
        
def plot_examples(gen,epoch=None,idx=None,training_name=None,root_folder=config.DEFAULT_ROOT_DIR,low_res_folder="training_images",format=False,verbose=True):
    image_dir = root_folder.format(low_res_folder)
    files = os.listdir(image_dir)
    gen.eval()
    for file in files:
        image_name = image_dir + "/" + file
        image = Image.open(image_name) # default_images="validation_images/"
        with torch.no_grad():
            upscaled_img = gen(
                config.test_transform(image=np.asarray(image))['image'].unsqueeze(0).to(config.DEVICE)
            )
        if type(epoch)==type(None) and type(idx)==type(None):
            SAVE_DIR = config.SAVE_RESULT_FOLDER.format(config.MODEL_NAME,low_res_folder)
            if verbose:
                print("=========>Saving The Image Inside {}".format(SAVE_DIR))
            if not os.path.exists(SAVE_DIR): #
                os.makedirs(SAVE_DIR)            
            save_image(upscaled_img*0.5+0.5,f"{SAVE_DIR}/{file}")
        else:
            if type(training_name) == type(None):
                SAVE_DIR = config.SAVE_RESULT_FOLDER.format(config.MODEL_NAME,config.SAVE_DIRECTORY_DURING_TRAINING)
            else:
                SAVE_DIR = config.SAVE_RESULT_FOLDER.format(config.MODEL_NAME,config.FORMAT_SAVE_DIRECTORY_DURING_TRAINING.format(training_name))  
            if verbose:
                print("=========>Saving The Image Inside {}".format(SAVE_DIR))
            if not os.path.exists(SAVE_DIR): #
                os.makedirs(SAVE_DIR)            
            save_image(upscaled_img*0.5+0.5,f"{SAVE_DIR}/epoch_{epoch}_idx_{idx}_{file}")
    gen.train()
    
    
    
def test_models(gen,folder,save_folder,epoch=None,idx=None,folder_name="BSDS100",verbose=True):
    files = os.listdir(folder)
    SAVE_RESULT_FOLDER = save_folder
    gen.eval()
    for file_ in files:
        image_name = folder + "/" + file_
        image = Image.open(image_name) # default_images="validation_images/"
        with torch.no_grad():
            upscaled_img = gen(
                config.test_transform(image=np.asarray(image))['image'].unsqueeze(0).to(config.DEVICE)
            )
        if type(epoch)==type(None) and type(idx)==type(None):
            if verbose:
                print("=========>Saving The Image Inside {}".format(SAVE_RESULT_FOLDER))
            if not os.path.exists(SAVE_RESULT_FOLDER): #
                os.makedirs(SAVE_RESULT_FOLDER)            
            if not os.path.exists(SAVE_RESULT_FOLDER + "/" + folder_name):
                print("===========> Creating Folder ",SAVE_RESULT_FOLDER + "/" + folder_name)
                os.makedirs(SAVE_RESULT_FOLDER + "/" + folder_name)            
            save_image(upscaled_img*0.5+0.5,f"{SAVE_RESULT_FOLDER}/{folder_name}/{file_}")
    gen.train()

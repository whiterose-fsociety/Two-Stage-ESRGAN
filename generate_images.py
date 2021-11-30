import torch
import utils
from PIL import Image
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from model import Generator, Discriminator
from dataset import ImageFolder
import numpy as np
import cv2

gen = Generator(in_channels=3).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
g_optimizer = optim.Adam(gen.parameters(),     0.0001, (0.9, 0.999))  # Generator learning rate during adversarial network training.

if config.LOAD_ESRRESNET:
    utils.load_checkpoint(
    config.CHECKPOINT_ESRRESNET,
    gen,
    opt_gen,
    config.LEARNING_RATE,
    )

if config.LOAD_ESRGAN:
    utils.load_checkpoint(
    config.CHECKPOINT_GEN,
    gen,
    opt_gen,
    config.LEARNING_RATE,
    )


if config.LOAD_G_BEST:
    utils.load_checkpoint(
    config.SAVE_RESULT_FOLDER.format(config.MODEL_NAME,config.SAVE_WEIGHTS + "/" + config.CHECKPOINT_GEN_ADV),
    gen,
    g_optimizer,
    config.LEARNING_RATE,
    )
    
plot_examples(gen,epoch=None,idx=None,root_folder=config.DEFAULT_ROOT_DIR,low_res_folder=config.LR_TRAIN_IMAGES,verbose=True)
plot_examples(gen,epoch=None,idx=None,root_folder=config.DEFAULT_ROOT_DIR,low_res_folder=config.LR_VALID_IMAGES,verbose=True)
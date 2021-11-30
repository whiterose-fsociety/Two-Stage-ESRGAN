import torch
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
import model
import h2lmodel
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch import nn
import os
from torch.utils.tensorboard import SummaryWriter

"""
================================= COMMENT THE ONE DEPENDENT ON THE OTHER  =================================
"""

EXPERIMENT_NUMBER = 8
EXP_NAME         = "8"
WEIGHT_EXPERIMENT = "Saved Weights {}".format(EXPERIMENT_NUMBER)
MODEL_NAME = "Proposed Combined Enhanced Super-Resolution Generative Adversarial Network/{}".format(WEIGHT_EXPERIMENT)
SAVE_RESULT_FOLDER = "../Results/{}/{}"
SAVE_DIRECTORY_DURING_TRAINING = "during_training_images" # These images are saved after ith epoch
FORMAT_SAVE_DIRECTORY_DURING_TRAINING = "during_training_images/{}" # These images are saved after ith epoch
SAVE_WEIGHTS = "weights"
SAVE_WRITER  = SAVE_RESULT_FOLDER.format(MODEL_NAME,'summary') + "/{}"
SAVE_RESULTS = SAVE_RESULT_FOLDER.format(MODEL_NAME,'results') + "/{}"
SAVE_SAMPLES = SAVE_RESULT_FOLDER.format(MODEL_NAME,'samples') + "/{}"
writer = SummaryWriter(SAVE_WRITER.format("logs"))

"""
================================= LEARNED LR IMAGES =================================
"""
#EXPERIMENT_NUMBER = "Learned LR Images"
#WEIGHT_EXPERIMENT = "Saved Weights {}".format(EXPERIMENT_NUMBER)
#MODEL_NAME = "Enhanced Super-Resolution Generative Adversarial Network Without Penalty/{}".format(WEIGHT_EXPERIMENT)
#SAVE_RESULT_FOLDER = "../Results/{}/{}"
#SAVE_DIRECTORY_DURING_TRAINING = "during_training_images" # These images are saved after ith epoch
#SAVE_WEIGHTS = "weights"



# LOAD_MODEL = True
LOAD_ESRRESNET = False # Remember to make false after training the GAN
LOAD_ESRGAN = False
LOAD_P_BEST = False
LOAD_G_BEST = False
SAVE_MODEL = True
CHECKPOINT_ESRRESNET = "g-best.pth" # We are going to do the exact same process for the high to low - ESRGAN
CHECKPOINT_GEN = "g-best.pth"  #- Only for the gan
CHECKPOINT_GEN_ADV = "adv_gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4 # 0.003
NUM_EPOCHS = 1409
BATCH_SIZE= 1
NUM_WORKERS = 1
HIGH_RES = 128
LOW_RES = HIGH_RES//4
IMG_CHANNELS = 3
DEFAULT_ROOT_DIR = "../Images/{}" # Remember the format
LR_TRAIN_IMAGES = 'lr_training_images'
LR_VALID_IMAGES = 'lr_validation_images'



START_P_EPOCH = 0
START_EPOCH = 0
P_EPOCHS  = 200
EPOCHS = 1409

#============== CRITERIONS 
PSNR_CRITERION = nn.MSELoss().to(DEVICE)
PIXEL_CRITERION = nn.L1Loss().to(DEVICE)
CONTENT_CRITERION = model.ContentLoss().to(DEVICE)
ADVERSARIAL_CRITERION = nn.BCELoss().to(DEVICE)
bce = nn.BCEWithLogitsLoss().to(DEVICE)
PIXEL_WEIGHT = 0.01
CONTENT_WEIGHT = 1.0
ADVERSARIAL_WEIGHT = 0.005
mse = nn.MSELoss()


gen = model.Generator(in_channels=3).to(DEVICE)
disc = model.Discriminator(in_channels=3).to(DEVICE)


gen_h2l = h2lmodel.Generator(HIGH_RES).to(DEVICE)
disc_h2l = h2lmodel.Discriminator(LOW_RES).to(DEVICE)
H2L_LEARNING_RATE = 1e-4
ALPHA, BETA = 1, 0.05

#============== OPTIMIZER.
p_optimizer           = optim.Adam(gen.parameters(),     0.0002, (0.9, 0.999))  # Generator model learning rate during generator training.
d_optimizer           = optim.Adam(disc.parameters(), 0.0001, (0.9, 0.999))  # Discriminator learning rate during adversarial network training.
g_optimizer           = optim.Adam(gen.parameters(),     0.0001, (0.9, 0.999))  # Generator learning rate during adversarial network training.

# Scheduler.
milestones            = [EPOCHS * 0.125, EPOCHS * 0.250, EPOCHS * 0.500, EPOCHS * 0.750]
p_scheduler           = CosineAnnealingLR(p_optimizer, P_EPOCHS // 4, 1e-7)               # Generator model scheduler during generator training.
d_scheduler           = MultiStepLR(d_optimizer, list(map(int, milestones)), 0.5)         # Discriminator model scheduler during adversarial training.
g_scheduler           = MultiStepLR(g_optimizer, list(map(int, milestones)), 0.5)         # Generator model scheduler during adversarial training.



#============== H2L OPTIMIZER.
optim_D_h2l = optim.Adam(filter(lambda p: p.requires_grad, disc_h2l.parameters()), lr=H2L_LEARNING_RATE, betas=(0.0, 0.9))
optim_G_h2l = optim.Adam(gen_h2l.parameters(), lr=H2L_LEARNING_RATE, betas=(0.0, 0.9))
h2l_d_scheduler           = MultiStepLR(optim_D_h2l, list(map(int, milestones)), 0.5)         # Discriminator model scheduler during adversarial training.
h2l_g_scheduler           = MultiStepLR(optim_G_h2l, list(map(int, milestones)), 0.5)         # Generator model scheduler during adversarial training.








# Additional variables.
EXP_DIR1              = os.path.join(SAVE_SAMPLES.format("sample"), EXP_NAME)
EXP_DIR2              = os.path.join(SAVE_RESULTS.format("result"), EXP_NAME)


#LR_TRAIN_IMAGES = 'learned_lr_training_images'
#LR_VALID_IMAGES = 'learned_lr_validation_images'

highres_transform = A.Compose(
    [A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),ToTensorV2()]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)


hr_transforms = A.Compose(
    [A.Normalize(mean=[0,0,0],std=[1,1,1]),ToTensorV2()]
)
lr_transforms = A.Compose(
    [A.Normalize(mean=[0,0,0],std=[1,1,1]),ToTensorV2()]
)



lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)
# trans = transforms.tworandom(transforms.compose([...]))
# dataset = dset.SegmentationDset(...,transform=trans, target_transform=trans)
# https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914
both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

import torch
import config
from torch import nn
from torch import optim
from utils import gradient_penalty, load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator, initialize_weights,test
# from tqdm import tqdm
# from random_dataset import ImageFolder
from dataset import ImageFolder
import utils
# from torch.utils.tensorboard import SummaryWriter
# from train import train_fn
from train import *
print("===========> We are training on the {}".format(DEVICE))
torch.cuda.empty_cache()
dir_name="train"
dataset = ImageFolder(dir_name=dir_name)
valid_dataset = ImageFolder(dir_name="valid")
loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=config.NUM_WORKERS,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=config.NUM_WORKERS,
)
torch.backends.cudnn.benchmark = True
best_psnr_value = 0.0

if config.LOAD_ESRRESNET:
    utils.load_checkpoint(config.CHECKPOINT_GEN,config.gen,config.p_optimizer,config.LEARNING_RATE)

if config.LOAD_ESRGAN:
    utils.load_checkpoint(
    config.CHECKPOINT_GEN_ADV,
    config.gen,
    config.g_optimizer,
    config.LEARNING_RATE,
    )
    utils.load_checkpoint(
    config.CHECKPOINT_DISC, config.disc, config.d_optimizer, config.LEARNING_RATE,
    )

if config.LOAD_P_BEST:
    print("==========>Loading the latest model from {}".format(os.path.join(EXP_DIR2, "p-best.pth")))
    gen.load_state_dict(torch.load(os.path.join(EXP_DIR2, "p-best.pth")))
if not os.path.exists(config.EXP_DIR1):
    os.makedirs(config.EXP_DIR1)
    
if not os.path.exists(config.EXP_DIR2):
    os.makedirs(config.EXP_DIR2)

# for epoch in range(config.START_P_EPOCH,config.P_EPOCHS):
#     train_generator(loader,epoch)
#     psnr_value = validate(valid_loader,epoch,"generator")
#     is_best = psnr_value > best_psnr_value
#     best_psnr_value = max(psnr_value,best_psnr_value)
#     if config.SAVE_MODEL:
#         save_checkpoint(gen, p_optimizer, filename=config.CHECKPOINT_GEN)
#     if is_best:
#         torch.save(gen.state_dict(),os.path.join(config.EXP_DIR2,"p-best.pth"))
#     p_scheduler.step()

# torch.save(gen.state_dict(),os.path.join(config.EXP_DIR2,"p-best.pth"))
# best_psnr_value = 0.0
# gen.load_state_dict(torch.load(os.path.join(EXP_DIR2, "p-best.pth")))
for epoch in range(config.START_EPOCH,config.P_EPOCHS):
    # Train each epoch for adversarial network.
    train_adversarial(loader, epoch)
    psnr_value = validate(valid_loader, epoch, "adversarial")
    is_best = psnr_value > best_psnr_value
    best_psnr_value = max(psnr_value, best_psnr_value)
    if config.SAVE_MODEL:
        save_checkpoint(gen, g_optimizer, filename=config.CHECKPOINT_GEN_ADV)
        save_checkpoint(disc, d_optimizer, filename=config.CHECKPOINT_DISC)
    if is_best:
        torch.save(disc.state_dict(),os.path.join(config.EXP_DIR2,"d-best.pth"))
        torch.save(gen.state_dict(),os.path.join(config.EXP_DIR2,"g-best.pth"))
    # Adjust the learning rate of the adversarial model.
    d_scheduler.step()
    g_scheduler.step()
torch.save(disc.state_dict(),os.path.join(config.EXP_DIR2,"d-best.pth"))
torch.save(gen.state_dict(),os.path.join(config.EXP_DIR2,"g-best.pth"))
import torch
import config
from config import *
from torch import nn
from torch import optim
from utils import gradient_penalty, load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import utils
import numpy as np
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
# gen = model.Generator(in_channels=3).to(DEVICE) # SRGAN
# disc = model.Discriminator(in_channels=3).to(DEVICE) # SRGAN


# gen_h2l = h2lmodel.Generator(HIGH_RES).to(DEVICE) # High To Low
# disc_h2l = h2lmodel.Discriminator(LOW_RES).to(DEVICE) # High To Low
def train_adversarial(loader,epoch):
    batches = len(loader)
    disc.train()
    gen.train()
    gen_h2l.train()
    disc_h2l.train()
    for index,(lr,hr,hr_down) in enumerate(loader):
        # print("================== Hey man we are here {}".format(index))
        optim_D_h2l.zero_grad()
        optim_G_h2l.zero_grad()
        d_optimizer.zero_grad()
        

        lr = lr.to(config.DEVICE)
        hr = hr.to(config.DEVICE)
        hr_down = hr_down.to(config.DEVICE)
        zs = np.random.randn(config.BATCH_SIZE, 1, config.HIGH_RES).astype(np.float32)
        zs= torch.from_numpy(zs).to(config.DEVICE)
        label_size = lr.size(0)

        lr_gen = gen_h2l(hr,zs) #fake_low_res Fake Low Res
        lr_gen_detach = lr_gen.detach() 
        sr = gen(lr_gen) #fake =  Fake High Res
        # print("LR GEN",lr_gen)
        # print("========================")
        # print("SR",sr)
        # print("========================")
        # print("SR SHAPE: ",sr.shape,"MIN",sr.amin(),"MAX",sr.amax())
        # print("LR SHAPE: ",lr.shape,"MIN",lr.amin(),"MAX",lr.amax())
        # print("LR GEN SHAPE: ",lr_gen.shape,"MIN",lr_gen.amin(),"MAX",lr_gen.amax())


        real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=DEVICE)
        fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=DEVICE)

        # Initialize the gradient of the discriminator model.
        
        # Generate super-resolution images.
        # sr = gen(lr)
        
        # Calculate the loss of the discriminator model on the high-resolution image.
        hr_output = disc(hr)
        sr_output = disc(sr.detach())
        # print("HR_OUTPUT: ",hr_output.shape,"Min ",hr_output.amin(),"Max ",hr_output.amax())
        # print("SR_OUTPUT: ",sr_output.shape,"Min ",sr_output.amin(),"Max ",sr_output.amax())
        # print("REAL LABEL: ",real_label.shape,"Min ",real_label.amin(),"Max ",real_label.amax())
        # print("First Argument of the thing",hr_output - torch.mean(sr_output))
        clamp = torch.clamp(hr_output - torch.mean(sr_output), min=0, max=1) # Relativistic Discriminator
        d_loss_hr = ADVERSARIAL_CRITERION(clamp, real_label) 
        d_loss_hr.backward()
        
        disc_real_h2l = disc_h2l(lr) 
        disc_fake_h2l = disc_h2l(lr_gen.detach())
        disc_loss_real_h2l = bce(disc_real_h2l,torch.ones_like(disc_real_h2l) - 0.1*torch.rand_like(disc_real_h2l))
        disc_loss_fake_h2l = bce(disc_fake_h2l,torch.zeros_like(disc_fake_h2l))
        loss_disc_h2l = (disc_loss_real_h2l + disc_loss_fake_h2l)/2 #CHANGE BACK TO THIS
        # loss_D_h2l = nn.ReLU()(1.0 - disc_h2l(lr)).mean() + nn.ReLU()(1 + disc_h2l(lr_gen_detach)).mean()
        
        # loss_disc.backward()
        

        
        d_hr = hr_output.mean().item()
        # Calculate the loss of the discriminator model on the super-resolution image.
        hr_output = disc(hr)
        sr_output = disc(sr.detach())
        # print("HR_OUTPUT: ",hr_output.shape,"Min ",hr_output.amin(),"Max ",hr_output.amax())
        # print("SR_OUTPUT: ",sr_output.shape,"Min ",sr_output.amin(),"Max ",sr_output.amax())
        # print("FAKE LABEL: ",fake_label.shape,"Min ",fake_label.amin(),"Max ",fake_label.amax())
        # print("First Argument of the thing",hr_output - torch.mean(sr_output))
        clamp_sr = torch.clamp(sr_output - torch.mean(hr_output), min=0, max=1) # # Relativistic Discriminator
        d_loss_sr = ADVERSARIAL_CRITERION(clamp_sr, fake_label)
        d_loss_sr.backward()
        d_sr1 = sr_output.mean().item()
        # Update the weights of the discriminator model.
        d_loss = d_loss_hr + d_loss_sr

        # loss_D_h2l.backward()
        loss_disc_h2l.backward()
        
        optim_D_h2l.step()
        d_optimizer.step()
        
        optim_D_h2l.zero_grad()
        # disc_fake = disc
        # sr = gen(lr_gen)
        # Initialize the gradient of the generator model.
        # sr_output = disc(sr)
        # disc_h2l
        gan_loss_h2l = -disc_h2l(lr_gen).mean()
        mse_loss_h2l = mse(lr_gen, hr_down)
        loss_G_h2l = ALPHA * mse_loss_h2l + BETA * gan_loss_h2l
        loss_G_h2l.backward()
        optim_G_h2l.step()
        d_optimizer.zero_grad()
        # Generate super-resolution images.
        # sr = gen(lr)
        
        # Calculate the loss of the discriminator model on the super-resolution image.
        g_optimizer.zero_grad()
        # sr = gen(lr)
        sr = gen(lr_gen_detach)
        hr_output = disc(hr.detach())
        sr_output = disc(sr.detach())
        # Perceptual loss=0.01 * pixel loss + 1.0 * content loss + 0.005 * adversarial loss.
        pixel_loss = PIXEL_WEIGHT * PIXEL_CRITERION(sr, hr.detach())
        content_loss = CONTENT_WEIGHT * CONTENT_CRITERION(sr, hr.detach())
        clamp_adversarial_loss = torch.clamp(sr_output - torch.mean(hr_output),min=0,max=1)
        adversarial_loss = ADVERSARIAL_WEIGHT * ADVERSARIAL_CRITERION(clamp_adversarial_loss, real_label)
        # Update the weights of the generator model.
        g_loss = pixel_loss + content_loss + adversarial_loss

        g_loss.backward()
        g_optimizer.step()
        d_sr2 = sr_output.mean().item()

        # Write the loss during training into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar(config.SAVE_WRITER.format("Train_Adversarial_H2L/G_Loss"),loss_G_h2l.item(),iters)
        writer.add_scalar(config.SAVE_WRITER.format("Train_Adversarial_H2L/D_Loss"),loss_disc_h2l.item(),iters)
        writer.add_scalar(config.SAVE_WRITER.format("Train_Adversarial/D_Loss"),d_loss.item(),iters)
        writer.add_scalar(config.SAVE_WRITER.format("Train_Adversarial/G_Loss"),g_loss.item(),iters)
        writer.add_scalar(config.SAVE_WRITER.format("Train_Adversarial/D_HR"),d_hr,iters)
        writer.add_scalar(config.SAVE_WRITER.format("Train_Adversarial/D_SR1"),d_sr1,iters)
        writer.add_scalar(config.SAVE_WRITER.format("Train_Adversarial/D_SR2"),d_sr2,iters)
        # utils.plot_examples(gen,epoch,idx,root_folder=config.DEFAULT_ROOT_DIR,low_res_folder=config.LR_TRAIN_IMAGES)
        if (index+1) % 100 == 0 and index > 0:
            utils.plot_examples(gen,epoch,index,root_folder=config.DEFAULT_ROOT_DIR,low_res_folder=config.LR_TRAIN_IMAGES)
            print(f"Train stage: adversarial "
            f"Epoch[{epoch + 1:04d}/{EPOCHS:04d}]({index + 1:05d}/{batches:05d}) "
            f"D Loss: {d_loss.item():.6f} G Loss: {g_loss.item():.6f} "
            f"D(HR): {d_hr:.6f} D(SR1)/D(SR2): {d_sr1:.6f}/{d_sr2:.6f}.")


def validate(valid_dataloader,epoch,stage):
    batches = len(valid_dataloader)
    gen.eval()
    total_psnr_value = 0.0
    with torch.no_grad():
        for index ,(lr,hr,_) in enumerate(valid_dataloader):
            lr = lr.to(DEVICE)
            hr = hr.to(DEVICE)
            sr = gen(lr)
            mse_loss = PSNR_CRITERION(sr,hr)
            psnr_value = 10*torch.log10(1/mse_loss).item()
            total_psnr_value +=psnr_value
        avg_psnr_value = total_psnr_value / batches
        if stage == "generator":
            writer.add_scalar(config.SAVE_WRITER.format("Val_Generator/PNSR"),avg_psnr_value,epoch+1)
        elif stage == "adversarial":
            writer.add_scalar(config.SAVE_WRITER.format("Val_Adversarial/PNSR"),avg_psnr_value,epoch+1)
        print(f"Valid stage: {stage} Epoch[{epoch + 1:04d}] avg PSNR: {avg_psnr_value:.2f}.\n")
    return avg_psnr_value

def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1,
    vgg_loss,
    g_scaler,
    d_scaler,
    tb_step,
    epoch
):
    # loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loader):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        # fake = gen(low_res)
        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            critic_real = disc(high_res)
            critic_fake = disc(fake.detach())
            gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
            )

        opt_disc.zero_grad()
        d_scaler.scale(loss_critic).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * l1(fake, high_res)
            adversarial_loss = 5e-3 * -torch.mean(disc(fake)) # Decrease slightly if model is stuck
            loss_for_vgg = vgg_loss(fake, high_res)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss
            
            
            # # l1_loss = 1e-2 * l1(fake, high_res)
            # adversarial_loss = 5e-3 * -torch.mean(disc(fake)) # Decrease slightly if model is stuck
            # loss_for_vgg = vgg_loss(fake, high_res)
            # gen_loss = l1_loss 

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward(retain_graph=True)
        g_scaler.step(opt_gen)
        g_scaler.update()

#         writer.add_scalar("Critic loss", loss_critic.item(), global_step=tb_step)
        tb_step += 1

#         if idx % 100 == 0 and idx > 0:
#             plot_examples("test_images/", gen)

        if idx % 100 == 0 and idx > 0:
            print(f"===>Saving Image At Epoch {epoch} with Batch Number {idx}")
            utils.plot_examples(gen,epoch,idx,root_folder=config.DEFAULT_ROOT_DIR,low_res_folder=config.LR_TRAIN_IMAGES)

            
#         loop.set_postfix(
#             gp=gp.item(),
#             critic=loss_critic.item(),
#             l1=l1_loss.item(),
#             vgg=loss_for_vgg.item(),
#             adversarial=adversarial_loss.item(),
#         )

    return tb_step
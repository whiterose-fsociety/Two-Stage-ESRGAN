import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
from spec_norm import SpectralNorm
HIGH_RES = 128
LOW_RES = HIGH_RES//4
# ========================================================= DISCRIMINATOR =========================================================
kwargs = {"kernel_size":3,"stride":1,"padding":1}
class ConvolutionalBlock(nn.Module):
    def __init__(self,in_channels,out_channels,downsample=False,initial=False,**kwargs):
        super().__init__()
        self.downsample = downsample
        self.initial = initial
        self.pooling = nn.Sequential(nn.AvgPool2d(2,2),SpectralNorm(nn.Conv2d(in_channels,out_channels,**kwargs,bias=False)))
        self.spectral = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels,out_channels,**kwargs,bias=False)))
        self.block = nn.Sequential(
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(in_channels,out_channels,**kwargs,bias=False)), 
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(out_channels,out_channels,**kwargs,bias=False)) if not self.downsample else nn.Sequential(nn.AvgPool2d(2,2),SpectralNorm(nn.Conv2d(out_channels,out_channels,**kwargs,bias=False)))
        )

        
                
    def forward(self,x):
        if self.initial:
            return self.block(x) + self.spectral(x)
        else:
            if not self.downsample:
                return self.block(x) + x
            else: # If Downsampling is true than perform skip connection and spec norm with average pooling
                initial = self.pooling(x)
                return self.block(x) + initial
        
        
        
class Discriminator(nn.Module):
    def __init__(self,input_size,in_channels=3):
        super().__init__()
        self.last_channels = (HIGH_RES // LOW_RES) if input_size == HIGH_RES else (HIGH_RES // LOW_RES)//2
        self.blocks = [in_channels,*([128]*2),*([256]*2),*([512]*self.last_channels)]
        self.disc_blocks = nn.Sequential(
        ConvolutionalBlock(self.blocks[0],self.blocks[1],initial=True,kernel_size=3,stride=1,padding=1),
        ConvolutionalBlock(self.blocks[1],self.blocks[2],initial=False,kernel_size=3,stride=1,padding=1)
        )
        self.downsample_blocks = nn.Sequential(
            *[ConvolutionalBlock(previous,current,initial=False,downsample=True,kernel_size=3,stride=1,padding=1) for (previous,current) in zip(self.blocks[2:],self.blocks[3:])]
        )
        self.classifier = nn.Sequential(
            SpectralNorm(nn.Linear(4*self.blocks[-1], self.blocks[-1])),
            nn.ReLU(),
            SpectralNorm(nn.Linear(self.blocks[-1], 1))
        )
            
    def forward(self,x):
        out = self.disc_blocks(x)
        out = self.downsample_blocks(out)
        out = out.view(-1,4*self.blocks[-1])
        out = self.classifier(out)
        return out
        


def discriminator_test():
    lr_noise = torch.randn(1,3,32,32)
    hr_noise = torch.randn(1,3,128,128)
    lr_discriminator = Discriminator(32)(lr_noise)
    hr_discriminator = Discriminator(128)(hr_noise)
    print("LR Discriminative: ",lr_discriminator)
    print("HR Discriminative: ",hr_discriminator)

    
    
# ========================================================= GENERATOR =========================================================
    
class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,pool=False,**kwargs):
        super().__init__()
        self.pool = pool
        self.convolve = nn.Sequential(nn.AvgPool2d(2,2),nn.Conv2d(in_channels,out_channels,**kwargs,bias=False))
        self.blocks = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels,out_channels,**kwargs,bias=False) if not self.pool else nn.Sequential(nn.AvgPool2d(2,2),nn.Conv2d(in_channels,out_channels,**kwargs,bias=False))
        )
        
    def forward(self,x):
        if not self.pool:
            return self.blocks(x)
        else:
            return self.blocks(x) + self.convolve(x)
        
        
class UpsampleBlock(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super().__init__()
        self.upscale = nn.PixelShuffle(2)
        self.blocks = nn.Sequential(*([BasicBlock(in_channels,out_channels,**kwargs)])*2)
        
    def forward(self,x):
        initial = self.upscale(x)
        return initial + self.blocks(initial)
        

class UpsampleGroup(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super().__init__()
        self.blocks = nn.Sequential(
            UpsampleBlock(in_channels,out_channels,**kwargs),
            BasicBlock(in_channels,in_channels,pool=False,**kwargs)
            )

    def forward(self,x):
        return self.blocks(x)
    
class ConvolutionalGroup(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super().__init__()
        self.blocks = nn.Sequential(
            BasicBlock(in_channels,in_channels,pool=False,**kwargs),
            BasicBlock(in_channels,out_channels,pool=True,**kwargs),
            BasicBlock(out_channels,out_channels,pool=False,**kwargs),
            BasicBlock(out_channels,out_channels,pool=False,**kwargs)
        )
    
    def forward(self,x):
        initial = self.blocks[:2](x)
        out = self.blocks(x)
        return out + initial
    
    
class Generator(nn.Module):
    def __init__(self,noise_dim=256):
        super().__init__()
        self.blocks = [96,96,128,256,512]
        self.up_blocks = [128,32]
        self.noise_dim = noise_dim
        self.noise_fc = nn.Linear(self.noise_dim,self.noise_dim*self.noise_dim)
        self.in_layer = nn.Conv2d(4, self.blocks[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample_blocks = nn.Sequential(*[ConvolutionalGroup(previous,current,**kwargs) for previous,current in zip(self.blocks,self.blocks[1:])])
        self.upsample_blocks = nn.Sequential(*[UpsampleGroup(block,block,**kwargs) for block in self.up_blocks])
        self.out_layer = nn.Sequential(
            nn.Conv2d(self.up_blocks[-1],8,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self,x,z):
        noise = self.noise_fc(z)
        noise = noise.view(-1,1,self.noise_dim,self.noise_dim)
        out = torch.cat((x,noise),1)
        out = self.in_layer(out)
        out = self.downsample_blocks(out)
        out = self.upsample_blocks(out)
        out = self.out_layer(out)
        return out
    
    
def generator_test(high_res=256):
    Z = np.random.randn(1, 1, high_res).astype(np.float32)
    Z = torch.from_numpy(Z)
    X = np.random.randn(1, 3, high_res, high_res).astype(np.float32)
    X = torch.from_numpy(X)
    generated_lr = Generator(high_res)(X,Z)
    print("Generated LR Image Shape: ",generated_lr.shape)
    
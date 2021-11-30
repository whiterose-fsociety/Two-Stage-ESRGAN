# Two-Stage-ESRGAN
Two-Stage ESRGAN

## Running
> Ensure that you have python3 installed

`$ python3 -m pip install torch torchvision torchaudio pytorch albumentations sklearn

`$ python3 esrgan.py
 
## Preparing The Database
> The dataset.py file

`$ Collect LR image with unknown degradation as well as corresponding HR image.

`$ Get random crop of size 256 x 256
  
  
 > The random_dataset.py file

`$ Collect LR image with unknown degradation as well as corresponding HR image.

`$ Randomize the order such that images LR-HR do not correspond

`$ Get random crop of size 256 x 256
  
 


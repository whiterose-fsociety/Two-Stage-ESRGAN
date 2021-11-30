# Two-Stage-ESRGAN
Two-Stage ESRGAN

## Running
> Ensure that you have python3 installed

`$ python3 -m pip install torch torchvision torchaudio pytorch albumentations sklearn

`$ sbatch slurm

`$ virtualenv -p=/usr/bin/python3.8 venv`

`$ source venv/bin/activate`

`$ cd epiuse`

`$ pip install -r requirements.txt`
 
> If pip does not work try this

 `$ python3 -m pip install -r requirements.txt`
 
## Preparing The Database
> The dataset.py file

`$ Collect LR image with unknown degradation as well as corresponding HR image.

`$ Get random crop of size 256 x 256
  
  
 > The dataset.py file

`$ Collect LR image with unknown degradation as well as corresponding HR image.

`$ Randomize the order such that images LR-HR do not correspond

`$ Get random crop of size 256 x 256
  
 


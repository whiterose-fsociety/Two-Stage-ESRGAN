# Two-Stage-ESRGAN
Two-Stage ESRGAN

![3](https://user-images.githubusercontent.com/42980126/144039574-10f76cca-dd3c-4755-a86b-d2b56aa8e22c.png)

## Changing The Configurations

`Check the config.py`

## Training The Model
> Ensure that you have python3 installed

`$ python3 -m pip install torch torchvision torchaudio pytorch albumentations sklearn`

`$ python3 esrgan.py`

## Evaluating The Model
> Evaluate the models and generate images

`$ python3 generate_images.py`

> Perform image quality assessment

`$ python3 generate_results.py`
 
## Preparing The Database
> The dataset.py file

`Collect LR image with unknown degradation as well as corresponding HR image.`

`Get random crop of size 256 x 256`
  
  
 > The random_dataset.py file

`Collect LR image with unknown degradation as well as corresponding HR image.`

`Randomize the order such that images LR-HR do not correspond`

`Get random crop of size 256 x 256`
  
 


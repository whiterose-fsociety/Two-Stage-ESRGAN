import torch
import config
from config import *
from torch import nn
from torch import optim
import utils
import numpy as np
# value = torch.tensor([6.9182e-05])
value = torch.tensor([-7.1396e-05])
real_label = torch.tensor([1.0])
fake_label = torch.tensor([0.0])
print(ADVERSARIAL_CRITERION(value,real_label))
print(ADVERSARIAL_CRITERION(value,fake_label))

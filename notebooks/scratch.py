import sys
sys.path.append("/usr/local/lib/python3.6/site-packages/IPython/extensions")
import autoreload

import numpy as np
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2
%matplotlib inline

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from src.models.conv import *
import src.utils as utils

mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])
ds = datasets.MNIST('./data/', train=True, download=True, transform=mnist_transforms)

im, targ = ds[0]

plt.imshow(im.squeeze().numpy(), cmap='gray')

model = SimpleConvNet()
model

pred = model.forward(im.unsqueeze(0))
pred.shape

model = FullyConvolutional()
pred = model.forward(im.unsqueeze(0))
pred.shape
out_features = pred.view(pred.shape[0], -1).shape[1]
out_features

head = DenseHead(out_features, 10)
pred = head(pred)

model = nn.Sequential(model, head)

loaded = utils.load_checkpoint("models/mnist_fully_conv", best=True)
model.load_state_dict(loaded['model'])

test_ds = datasets.MNIST('./data/', train=False, download=True, transform=mnist_transforms)
print(len(test_ds))
loader = torch.utils.data.DataLoader(test_ds, batch_size=32)

preds = []
targs = []
inps = []
for im, targ in loader:
  pred = model.forward(im)
  preds.append(pred)
  inps.append(im)
  targs.append(targ)
raw_preds = torch.cat(preds).detach().numpy()
inps = torch.cat(inps).squeeze().numpy()
targs = torch.cat(targs).numpy()
preds = np.argmax(raw_preds, axis=1)

correct = np.where(preds == targs)[0]
incorrect = np.where(preds != targs)[0]

def foo():
  fig, axs = plt.subplots(2, 2)
  idx = 0
  for i, ax in enumerate(np.array(axs).reshape(-1)):
    j = incorrect[idx + i]
    ax.imshow(inps[j])
    ax.set_title(f"{preds[j]}, {targs[j]}")
    
foo()
    



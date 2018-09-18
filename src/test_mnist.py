import numpy as np
import argparse
import logging
from logging.config import fileConfig
from pathlib import Path

import cdsw

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from src.models.conv import SimpleConvNet
import src.utils as utils

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-name', type=str, default="")
    parser.add_argument('--log-path', type=str, default="")
    parser.add_argument('--log-file', type=str, default="")
    parser.add_argument('--model-path', type=str, default="")
    parser.add_argument('--model-file', type=str, default="")
    args = parser.parse_args()
    fileConfig("logging_config.ini")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    gpu = torch.device("cuda:0")
    cpu = torch.device("cpu")
    device = gpu if args.cuda else cpu

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST('./data/', train=False, download=True, transform=mnist_transforms)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)

    model = SimpleConvNet()
    if args.cuda:
        model = model.to(gpu)

    loaded = utils.load_checkpoint(args.model_path, checkpoint_file=args.model_file)
    model.load_state_dict(loaded['model'])

    # Log our parameters into mlflow
#    for key, value in vars(args).items():
#        mlflow.log_param(key, value)

    loss = 0.
    correct = 0.
    n = len(loader.dataset)
    model.eval()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        prediction = torch.argmax(output, dim=1)
        correct += torch.sum(prediction == target).item()
        loss += F.nll_loss(output, target).item()
    loss /= n
    acc = correct / n
    cdsw.track_metric("test_loss", loss)
    cdsw.track_metric("test_acc", acc)
    logging.debug('Test Loss: {:.6f} Test Accuracy: {:.4f}'.format(loss, acc))


import numpy as np
import argparse
import logging
from logging.config import fileConfig
from pathlib import Path

import mlflow

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from src.models.conv import SimpleConvNet
import src.utils as utils

if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-name', type=str, default="simple")
    parser.add_argument('--checkpoint-path', type=str, default=None)
    args = parser.parse_args()
    fileConfig("logging_config.ini")

    use_gpu = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")

    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
    mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST('./data/', train=False, download=True, transform=mnist_transforms)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)

    model = SimpleConvNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    loaded = loaded = utils.load_checkpoint(args.checkpoint_path, best=True)
    model.load_state_dict(loaded['model'])

    with mlflow.start_run():
        # Log our parameters into mlflow
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        loss = 0.
        correct = 0.
        n = len(loader.dataset)
        model.eval()
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            prediction = torch.argmax(output, dim=1)
            correct += torch.sum(prediction == target).item()
            loss += criterion(output, target).item()
        loss /= n
        acc = correct / n
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_acc", acc)
        logging.debug('Test Loss: {:.6f} Test Accuracy: {:.4f}'.format(loss, acc))


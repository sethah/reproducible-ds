import numpy as np
import argparse
import logging
from logging.config import fileConfig
from pathlib import Path

import mlflow
import mlflow.cli

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from src.models.conv import SimpleConvNet
import src.utils as utils


def train(epoch, model, loader, optimizer, device=torch.device("cpu"), log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            samples_processed = batch_idx * data.shape[0]
            total_samples = len(loader.sampler)
            logging.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, samples_processed, total_samples,
                100. * batch_idx / len(loader), loss.data.item()))
            step = epoch * len(loader) + batch_idx
            mlflow.log_metric('train_loss', loss.item())


def valid(epoch, model, loader, device):
    loss = 0.
    correct = 0.
    n = len(loader.sampler)
    model.eval()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        prediction = torch.argmax(output, dim=1)
        correct += torch.sum(prediction == target).item()
        loss += F.nll_loss(output, target).item()
    loss /= n
    acc = correct / n
    logging.debug('Train Epoch: {} Validation Loss: {:.6f} Validation Accuracy: {:.4f}'.format(
        epoch, loss, acc))
    return loss, acc


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-gamma', type=float, default=0.8)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--restore', type=str, default=None, help="{'best', 'latest'}")
    parser.add_argument('--checkpoint-path', type=str, default=None)

    args = parser.parse_args()
    fileConfig("logging_config.ini")

    use_gpu = args.gpu and torch.cuda.is_available()
    train_device = torch.device("cuda:0") if use_gpu else torch.device("cpu")

    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
    mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST('./data/', train=True, download=True, transform=mnist_transforms)

    idx = np.arange(len(ds))
    np.random.shuffle(idx)

    train_fraction = 0.8
    train_samples = int(train_fraction * len(ds))
    train_sampler = SubsetRandomSampler(idx[:train_samples])
    validation_sampler = SubsetRandomSampler(idx[train_samples:])

    train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                               sampler=train_sampler, **kwargs)

    validation_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                                    sampler=validation_sampler)

    model = SimpleConvNet().to(train_device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)

    best_loss = 1000000.
    if args.restore:
        loaded = utils.load_checkpoint(args.checkpoint_path, best=args.restore == 'best')
        model.load_state_dict(loaded['model'])
        optimizer.load_state_dict(loaded['optimizer'])
        lr_sched.load_state_dict(loaded['scheduler'])
        best_loss = loaded.get('best_loss', best_loss)

    with mlflow.start_run():
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        for epoch in range(1, args.epochs + 1):
            lr_sched.step()
            train(epoch, model, train_loader, optimizer, device=train_device,
                  log_interval=args.log_interval)
            loss, acc = valid(epoch, model, validation_loader, device=train_device)
            mlflow.log_metric('valid_loss', loss)
            mlflow.log_metric('valid_acc', acc)
            if args.checkpoint_path:
                save_dict = {'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'scheduler': lr_sched.state_dict(),
                             'best_loss': best_loss}
                utils.save_checkpoint(args.checkpoint_path, save_dict)
                if loss < best_loss:
                    best_loss = loss
                    utils.save_checkpoint(args.checkpoint_path, save_dict, best=True)

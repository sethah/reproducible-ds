import argparse
import logging
from pathlib import Path
import tempfile

import mlflow

import torch
import torch.nn.functional as F
import torch.optim as optim
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
            logging.log(logging.INFO,
                        'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                       100. * batch_idx / len(loader), loss.data.item()))
            step = epoch * len(loader) + batch_idx
            mlflow.log_metric('train_loss', loss.data.item())


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
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-name', type=str, default="")
    parser.add_argument('--log-path', type=str, default="")
    parser.add_argument('--log-file', type=str, default="")
    parser.add_argument('--restore', type=int, default=0)
    parser.add_argument('--checkpoint-path', type=str, default="")
    parser.add_argument('--checkpoint-file', type=str, default="")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    gpu = torch.device("cuda:0")
    cpu = torch.device("cpu")
    train_device = gpu if args.cuda else cpu

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    if args.log_path:
        log_file = Path(args.log_path) / f"{args.log_file}.log"
        fileHandler = logging.FileHandler(str(log_file))
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    rootLogger.addHandler(consoleHandler)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    mnist_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('./data/', train=True, download=True, transform=mnist_transforms)
    test_ds = datasets.MNIST('./data/', train=False, download=True, transform=mnist_transforms)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                              shuffle=False, **kwargs)

    model = SimpleConvNet()
    if args.cuda:
        model = model.to(gpu)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.restore:
        loaded = utils.load_checkpoint(args.checkpoint_path, args.checkpoint_file)
        model.load_state_dict(loaded['model'])
        optimizer.load_state_dict(loaded['optimizer'])
        state = loaded['state']
        params = loaded['params']
    else:
        params = vars(args)
        state = {'epoch': 1}

    with mlflow.start_run():
        # Log our parameters into mlflow
        for key, value in params.items():
            mlflow.log_param(key, value)
        if args.log_path:
            mlflow.log_artifact(str(log_file))

        for epoch in range(state['epoch'], args.epochs + 1):
            train(epoch, model, train_loader, optimizer, device=train_device,
                  log_interval=args.log_interval)
            state['epoch'] += 1
            if args.checkpoint_path:
                save_dict = {'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'state': state,
                             'params': params}
                utils.save_checkpoint(args.checkpoint_path, save_dict, model_name=args.model_name)
            # test(epoch)


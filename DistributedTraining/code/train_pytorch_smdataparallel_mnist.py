rom __future__ import print_function

import argparse
import os
import time

import smdistributed.dataparallel.torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Network definition
from model_def import Net

# Import SMDataParallel PyTorch Modules
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


class CUDANotFoundException(Exception):
    pass


dist.init_process_group()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and args.rank == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data) * args.world_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        if args.verbose:
            print("Batch", batch_idx, "from rank", args.rank)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="For displaying smdistributed.dataparallel-specific logs",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/tmp/data",
        help="Path for downloading " "the MNIST dataset",
    )

    args = parser.parse_args()
    args.world_size = dist.get_world_size()
    args.rank = rank = dist.get_rank()
    args.local_rank = local_rank = dist.get_local_rank()
    args.lr = 1.0
    args.batch_size //= args.world_size // 8
    args.batch_size = max(args.batch_size, 1)
    data_path = args.data_path

    if args.verbose:
        print(
            "Hello from rank",
            rank,
            "of local_rank",
            local_rank,
            "in world size of",
            args.world_size,
        )

    if not torch.cuda.is_available():
        raise CUDANotFoundException(
            "Must run smdistributed.dataparallel MNIST example on CUDA-capable devices."
        )

    torch.manual_seed(args.seed)

    device = torch.device("cuda")

    # select a single rank per node to download data
    is_first_local_rank = local_rank == 0
    if is_first_local_rank:
        train_dataset = datasets.MNIST(
            data_path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
    dist.barrier()  # prevent other ranks from accessing the data early
    if not is_first_local_rank:
        train_dataset = datasets.MNIST(
            data_path,
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )
    if rank == 0:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_path,
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.test_batch_size,
            shuffle=True,
        )

    model = DDP(Net().to(device))
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        if rank == 0:
            test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
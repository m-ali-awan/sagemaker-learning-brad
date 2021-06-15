
from __future__ import print_function

import os

import torch

# Network definition
from model_def import Net


def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the mnist model")
        model.load_state_dict(torch.load(f, map_location=device))
    return model
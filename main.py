#!/usr/bin/env  python
from model import MobileSal, MobileSalTranspose
from blocks import *
import utils
from ray import tune
import ray
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
import numpy as np
import torch
import torch.utils.data as data
import pretty_errors
import torch.nn as nn
import os
import sys
import argparse
import matplotlib.pyplot as plt

#  import torchsummary
pretty_errors.configure(
    separator_character="*",
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,
    display_link=True,
    lines_before=5,
    lines_after=2,
    line_color=pretty_errors.RED + "> " + pretty_errors.default_config.line_color,
    code_color="  " + pretty_errors.default_config.line_color,
    truncate_code=True,
    display_locals=True,
)


seed = 1234823
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plotloss(loss, fig_num=0):
    fig, ax = plt.subplots(nrows=3, ncols=1)
    pred = {}

    sal = []
    idr = []
    total = []

    for ep, data in enumerate(loss):
        sal.append(data[0])
        idr.append(data[1])
        total.append(data[2])

    ax[0].title.set_text(f"Sal Loss")
    ax[1].title.set_text(f"IDR Loss")
    ax[2].title.set_text(f"Total Loss")

    x = [ep for ep in range(len(loss))]
    ax[0].plot(x, sal, label=f"Sal")
    ax[1].plot(x, idr, label=f"IDR")
    ax[2].plot(x, total, label=f"Total")
    plt.show()
    #  plt.savefig(f"loss{fig_num}")


def eval(model, dataloader):

    for i_batch, data in enumerate(dataloader):
        rgb, depth, gt = data
        print(rgb.shape)
        preds, reconstructed = model(rgb, depth)
        # Preds is tuple of 5 feature maps

        fig, ax = plt.subplots(ncols=3, nrows=4)
        rgb, depth, gt = data

        for i in range(rgb.shape[0]):
            print(i)
            ax[0][i].imshow(preds[4][i].reshape(320, 320, 1).cpu().detach())
            ax[1][i].imshow(gt[i].reshape(320, 320, 1).cpu().detach())
            ax[2][i].imshow(depth[i].reshape(320, 320, 1).cpu().detach())
            rgb_img = rgb[i].cpu().detach().numpy()
            ax[3][i].imshow(rgb_img.transpose(1, 2, 0).astype(np.int))
        plt.show()


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def main(args: argparse.Namespace, use_checkpoint=True) -> None:
    # [t, c, n s]
    # ref section 4 of mobilenet_v2 paper
    mobile_net_settings = [
        [5, 16, 1, 1],
        [5, 32, 2, 1],
        [5, 64, 3, 1],
        [5, 96, 4, 1],
        [5, 320, 3, 1],
    ]
    """
    Sizes of feature maps:
        [batch_size, 16, 160, 160]
        [batch_size, 32, 80, 80]
        [batch_size, 64, 40, 40]
        [batch_size, 96, 20, 20]
        [batch_size, 320, 10, 10]
    """
    #############
    # Ray Setup
    #############

    #  config = {
    #  "lr": tune.uniform(0.00001, 0.1),
    #  "lambda": tune.uniform(0, 1),
    #  "beta1": tune.uniform(0, 1),
    #  "beta2": tune.uniform(0, 1),
    #  }

    #  current_best_params = [
    #  {
    #  "lr": 0.0001,
    #  "lambda": 0.3,
    #  "beta1": 0.9,
    #  "beta2": 0.99,
    #  }
    #  ]

    #  search_algo = HyperOptSearch(
    #  metric="train_loss",
    #  mode="min",
    #  points_to_evaluate=current_best_params,
    #  )

    #  scheduler = ASHAScheduler(max_t=200, grace_period=30)

    #  reporter = tune.CLIReporter(
    #  metric_columns=["train_loss"],
    #  parameter_columns=["lr", "lambda"],
    #  )

    ################
    # Torch Setup
    ################

    # Sanity Check Dataset
    nju2k_dataset = utils.NJU2K_Dataset(
        rgb_dir="./data/sanity_check/RGB/",
        depth_dir="./data/sanity_check/depth/",
        ground_truth_dir="./data/sanity_check/GT/",
    )

    # Check if there is already a state_dict available
    #  if use_checkpoint:
    checkpoint_path = (
        "/home/epiphyte/Documents/Research/efficient_3d_Det/finetune/model0900"
    )
    checkpoint = torch.load(checkpoint_path)

    # Load Datasets
    #  nju2k_dataset = utils.NJU2K_Dataset(
    #  rgb_dir="./data/NJU2K/RGB/",
    #  depth_dir="./data/NJU2K/depth/",
    #  ground_truth_dir="./data/NJU2K/GT/",
    #  )

    #  dut_test = utils.DUT_Dataset(
    #  rgb_dir="./data/DUT_test/RGB/",
    #  depth_dir="./data/DUT_test/depth/",
    #  ground_truth_dir="./data/DUT_test/GT/",
    #  )

    # Dataloaders
    dataloader = data.DataLoader(
        nju2k_dataset, batch_size=3, shuffle=True, pin_memory=True, num_workers=5
    )
    #  validate = data.DataLoader(
    #  dut_test, batch_size=10, shuffle=True, pin_memory=True, num_workers=2
    #  )

    # Init Network
    net = MobileSalTranspose(mobile_net_settings).to(device)
    # torchsummary.summary(net, [(3, 320, 320), (1, 320, 320)])

    net.load_state_dict(checkpoint["model_state_dict"])
    # net.apply(init_weights)

    solver = utils.Solver(net)
    del solver.optim
    solver.optim = torch.optim.Adam(
        solver.model.parameters(), lr=0.00005, betas=(0.9, 0.99), weight_decay=0.00001
    )

    #  experiment_analysis = tune.run(
    #  tune.with_parameters(
    #  utils.train,
    #  model=net,
    #  train_set=dataloader,
    #  validate_set=dataloader,
    #  epochs=200,
    #  ),
    #  resources_per_trial={"cpu": 8, "gpu": 1},
    #  config=config,
    #  num_samples=10,
    #  scheduler=scheduler,
    #  search_alg=search_algo,
    #  progress_reporter=reporter,
    #  checkpoint_at_end=True,
    #  local_dir="./checkpoints",
    #  mode="min",
    #  metric="train_loss",
    #  )
    loss = solver(train_set=dataloader, validate_set=dataloader, epochs=1000)
    plotloss(loss, 1)
    print("Done Training NJU2K Dataset, starting NLPR...")

    #  # Remove Datasets from memory
    #  del nju2k_dataset
    #  del dataloader

    #  nlpr = utils.NLPR(
    #  rgb_dir="./data/NLPR/RGB/",
    #  depth_dir="./data/NLPR/depth/",
    #  ground_truth_dir="./data/NLPR/GT/",
    #  )
    #  dataloader_nlpr = data.DataLoader(
    #  nlpr, batch_size=10, shuffle=True, pin_memory=True, num_workers=5
    #  )
    #  loss = solver(train_set=dataloader_nlpr, validate_set=validate, epochs=60)
    #  plotloss(loss, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileSal Implementation")
    parser.add_argument(
        "--checkpoint_dir", type=str, help="Directory of model checkpoints"
    )

    args = parser.parse_args()
    main(args)

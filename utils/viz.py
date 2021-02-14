import torch
from torch import nn
import torch.nn.functional as F
import torchviz
from .pytorch_msssim import SSIM, ssim
import matplotlib.pyplot as plt
import numpy as np


def graph_checkpoint(
    preds, restored_dep, depth_img, rgb_img, gt, len_batch, loss_fn, dice_loss, epoch, run_iter
):
    fig, ax = plt.subplots(ncols=len(preds) + 4, nrows=len_batch - 1)

    for j in range(len_batch - 1):
        for i in range(len(preds)):
            bcee = str(round(loss_fn(preds[i][j], gt[j]).item(), 3))
            dicee = str(round(dice_loss(preds[i][j], gt[j]).item(), 3))

            ax[j][i].set(xlabel="BCE: " + bcee + "\nDice: " + dicee)
            ax[j][i].axes.tick_params(axis="both", which="both", length=0)
            ax[j][i].axes.tick_params(axis="both", which="both", length=0)
            ax[j][i].axes.tick_params(axis="both", which="both", length=0)
            ax[j][i].axes.tick_params(axis="both", which="both", length=0)
            ax[j][i].axes.tick_params(axis="both", which="both", length=0)
            ax[j][i].axes.tick_params(axis="both", which="both", length=0)
            ax[j][i].axes.get_yaxis().set_visible(False)
            ax[j][i].axes.get_yaxis().set_visible(False)
            ax[j][i].axes.get_yaxis().set_visible(False)
            ax[j][i].axes.get_yaxis().set_visible(False)
            ax[j][i].axes.get_yaxis().set_visible(False)
            ax[j][i].axes.get_yaxis().set_visible(False)

    for i in range(len_batch - 1):
        ax[i][0].imshow(preds[0][i].reshape(320, 320, 1).cpu().detach())
        ax[i][1].imshow(preds[1][i].reshape(320, 320, 1).cpu().detach())
        ax[i][2].imshow(preds[2][i].reshape(320, 320, 1).cpu().detach())
        ax[i][3].imshow(preds[3][i].reshape(320, 320, 1).cpu().detach())
        ax[i][4].imshow(preds[4][i].reshape(320, 320, 1).cpu().detach())
        ax[i][5].imshow(restored_dep[i].reshape(320, 320, 1).cpu().detach())
        ax[i][6].imshow(depth_img[i].reshape(320, 320, 1).cpu().detach())
        ax[i][7].imshow(gt[i].reshape(320, 320, 1).cpu().detach())
        rgb = rgb_img[i].cpu().detach().numpy()
        ax[i][8].imshow(rgb.transpose(1, 2, 0).astype(np.int))

    plt.savefig(f"valid{epoch}{run_iter}")

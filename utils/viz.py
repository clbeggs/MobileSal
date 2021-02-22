import torch
from torch import nn
import torch.nn.functional as F
import torchviz
from .pytorch_msssim import SSIM, ssim
import matplotlib.pyplot as plt
import numpy as np


def graph_checkpoint(
    preds,
    restored_dep,
    depth_img,
    rgb_img,
    gt,
    len_batch,
    loss_fn,
    dice_loss,
    idr_loss,
    epoch,
    run_iter,
):
    fig, ax = plt.subplots(ncols=len(preds) + 4, nrows=len_batch - 1)
    fig.set_size_inches(25, 20)

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
        ax[i][0].title.set_text("Pred 1")
        ax[i][1].title.set_text("Pred 2 ")
        ax[i][2].title.set_text("Pred 3 ")
        ax[i][3].title.set_text("Pred 4 ")
        ax[i][4].title.set_text("Pred 5 ")
        ax[i][5].title.set_text("Restored Depth")
        ax[i][6].title.set_text("Input Depth")
        ax[i][7].title.set_text("Ground Truth")
        ax[i][8].title.set_text("Input RGB")

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

        idr_loss_val = (
            1
            - idr_loss(
                restored_dep[i].reshape(1, 1, 320, 320), gt[i].reshape(1, 1, 320, 320)
            ).item()
        )
        ax[i][5].set(xlabel="IDR Loss: " + str(np.round(idr_loss_val, 4)))

    plt.savefig(f"valid{epoch}{run_iter}", dpi=192)
    #  plt.show()

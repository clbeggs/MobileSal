#  from kornia.losses import SSIMLoss
import torch
from torch import nn
import torch.nn.functional as F
from .pytorch_msssim import SSIM, ssim
from .viz import graph_checkpoint
import numpy as np
import visdom
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import visdom
from model import MobileSal, MobileSalTranspose


# torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  viz = visdom.Visdom()
#  validation_loss_plot = viz.scatter(np.array([[0, 0]]), opts=dict(title="Validation"))
#  training_loss_plot = viz.scatter(np.array([[0, 0]]), opts=dict(title="Training"))

run_iter = 0


#  def live_plot(epoch, datapoint, plot="valid"):
#  if plot == "valid":
#  viz.scatter(
#  X=np.array([[epoch, datapoint.cpu().detach().numpy()]]),
#  win=validation_loss_plot,
#  update="append",
#  )
#  else:
#  viz.scatter(
#  X=np.array([[epoch, datapoint.cpu().detach().numpy()]]),
#  win=training_loss_plot,
#  update="append",
#  )


def printweights(m):
    if isinstance(m, (nn.Conv2d)):
        print(m.weight)
    for child in m.children():
        printweights(child)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        #  inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


def save_model_dict(ep, loss, config, model):
    global run_iter
    torch.save(
        {
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": config["optim"].state_dict(),
            "loss": loss,
        },
        "./checkpoints/model{}{}".format(run_iter, ep),
    )


def validate(
    model,
    validate_set: torch.utils.data.DataLoader,
    epoch: int,
    config: dict,
    save_model: bool = False,
    num_to_test: int = 10,
) -> float:
    """Validate training
    Args:
        validate_set: Validation set
        epoch: Latest epoch
        save_model: Bool to force saving the model
    """
    num_processed = 0
    loss = 0
    loss_fn = config["loss_fn"]
    dice_loss = config["dice_loss"]
    idr_loss_fn = config["idr_loss_fn"]
    lam = config["lambda"]
    valid_loss = config["valid_loss"]
    preds, restored_dep, depth_img, rgb_img, gt = None, None, None, None, None

    for i_batch, sample_batched in enumerate(validate_set):
        rgb_img, depth_img, gt = sample_batched
        rgb_img = rgb_img.to(device)
        depth_img = depth_img.to(device)
        gt = gt.to(device)

        with torch.no_grad():

            # Get predictions and restored depth map
            preds, restored_dep = model(rgb_img, depth_img)

            # Salient loss for predictions, BCE(pred) + Dice(pred)
            sal_loss = loss_fn(preds[0], gt) + dice_loss(preds[0], gt)
            for j in range(1, len(preds)):
                sal_loss += loss_fn(preds[j], gt) + dice_loss(preds[j], gt)

            # Restored Depth map loss
            idr_loss = 1 - idr_loss_fn(restored_dep, gt)
            # Total loss and optimize
            loss = sal_loss + lam * idr_loss
            valid_loss.append([sal_loss, idr_loss, loss])
            print(f"Epoch: {epoch} Validation Loss: {loss}\n\n")
            #  live_plot(epoch, loss)
            num_processed += rgb_img.shape[0]

            if num_processed >= num_to_test:
                break

    len_batch = len(sample_batched)
    if (epoch % 50 == 0) or (save_model is True):
        #  global run_iter
        #  graph_checkpoint(
        #  preds,
        #  restored_dep,
        #  depth_img,
        #  rgb_img,
        #  gt,
        #  len_batch,
        #  loss_fn,
        #  dice_loss,
        #  idr_loss_fn,
        #  epoch,
        #  run_iter,
        #  )
        #  save_model_dict(ep=epoch, loss=loss, model=model, config=config)
        return loss
    else:
        return loss


def train(
    config,
    model,
    train_set: torch.utils.data.DataLoader,
    validate_set: torch.utils.data.DataLoader,
    epochs: int,
):
    """Single training run, does one pass through the dataloader
    Args:
        dataloader: torch.utils.data.DataLoader

    Returns:
        None
    """
    lam = config["lambda"]

    optim = torch.optim.Adam(
        model.parameters(),
        weight_decay=0.001,
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
    )

    lr_schedule = torch.optim.lr_scheduler.StepLR(
        optim, step_size=100, gamma=0.5, verbose=True
    )

    loss_fn = nn.BCELoss()
    dice_loss = DiceLoss()
    idr_loss_fn = SSIM()

    train_config = {
        "loss_fn": loss_fn,
        "dice_loss": dice_loss,
        "idr_loss_fn": idr_loss_fn,
        "lambda": lam,
        "optim": optim,
        "lr_schedule": lr_schedule,
        "valid_loss": [],
    }
    num_to_test = 10

    for epoch in range(epochs):
        len_batch = len(train_set)
        train_loss = None
        for i_batch, sample_batched in enumerate(train_set):
            print(f"Batch: [{i_batch}/{len_batch}]")

            rgb_img, depth_img, gt = sample_batched
            rgb_img = rgb_img.to(device)
            depth_img = depth_img.to(device)
            gt = gt.to(device)

            # Analagous to .zero_grad(), but faster
            for param in model.parameters():
                param.grad = None

            # Get predictions and restored depth map
            preds, restored_dep = model(rgb_img, depth_img)

            # Salient loss for predictions, BCE(pred) + Dice(pred)
            sal_loss = loss_fn(preds[0], gt) + dice_loss(preds[0], gt)
            for j in range(1, len(preds)):
                sal_loss += loss_fn(preds[j], gt) + dice_loss(preds[j], gt)

            # Restored Depth map loss
            idr_loss = 1 - idr_loss_fn(restored_dep, gt)

            # Total loss and optimize
            train_loss = sal_loss + lam * idr_loss
            train_loss.backward()
            optim.step()

        valid_loss = validate(
            model=model,
            validate_set=validate_set,
            epoch=epoch,
            config=train_config,
            save_model=False,
            num_to_test=num_to_test,
        )
        #  live_plot(epoch, train_loss, "train")
        tune.report(train_loss=train_loss.detach().cpu().numpy())
        lr_schedule.step()

    valid_loss = validate(
        model=model,
        validate_set=validate_set,
        epoch=epoch,
        config=train_config,
        save_model=True,
        num_to_test=num_to_test,
    )
    global run_iter
    run_iter += 1
    return train_config["valid_loss"]

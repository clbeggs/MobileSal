#  from kornia.losses import SSIMLoss
import torch
from torch import nn
import torch.nn.functional as F
from .pytorch_msssim import SSIM, ssim
import matplotlib.pyplot as plt
from .viz import graph_checkpoint
import numpy as np

# torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class Solver:
    def __init__(self, model):
        self.model = model
        self.dice_loss = DiceLoss()
        self.loss_fn = nn.BCELoss()
        self.idr_loss_fn = SSIM()
        self.lam = 0.3
        self.run_iter = 0
        self.optim = torch.optim.Adam(
            model.parameters(),
            weight_decay=0.0001,
            lr=0.0001,
            betas=(0.9, 0.99),
        )
        self.lr_schedule = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=10, gamma=0.5, verbose=True
        )
        self.valid_loss = []

    def save_model(self, ep, loss):
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "loss": loss,
            },
            "./checkpoints/model{}{}".format(self.run_iter, ep),
        )

    def validate(
        self,
        validate_set: torch.utils.data.DataLoader,
        epoch: int,
        save_model: bool = False,
        num_to_test: int = 10,
    ) -> None:
        """Validate training
        Args:
            validate_set: Validation set
            epoch: Latest epoch
            save_model: Bool to force saving the model
        """
        num_processed = 0
        for i_batch, sample_batched in enumerate(validate_set):
            rgb_img, depth_img, gt = sample_batched
            rgb_img = rgb_img.to(device)
            depth_img = depth_img.to(device)
            gt = gt.to(device)

            with torch.no_grad():

                # Get predictions and restored depth map
                preds, restored_dep = self.model(rgb_img, depth_img)

                # Salient loss for predictions, BCE(pred) + Dice(pred)
                sal_loss = self.loss_fn(preds[0], gt) + self.dice_loss(preds[0], gt)
                for j in range(1, len(preds)):
                    sal_loss += self.loss_fn(preds[j], gt) + self.dice_loss(
                        preds[j], gt
                    )

                # Restored Depth map loss
                idr_loss = 1 - self.idr_loss_fn(restored_dep, gt)
                # Total loss and optimize
                loss = sal_loss + self.lam * idr_loss
                self.valid_loss.append([sal_loss, idr_loss, loss])
                print(f"Epoch: {epoch} Total Loss: {loss}\n\n")
                num_processed += rgb_img.shape[0]

                if num_processed >= num_to_test:
                    break

        len_batch = len(sample_batched)
        if (epoch % 10 == 0) or (save_model is True):
            graph_checkpoint(
                preds,
                restored_dep,
                depth_img,
                rgb_img,
                gt,
                len_batch,
                self.loss_fn,
                self.dice_loss,
                epoch,
                self.run_iter,
            )
            self.save_model(ep=epoch, loss=loss)
            return 0

    def run_solver(
        self,
        train_set: torch.utils.data.DataLoader,
        validate_set: torch.utils.data.DataLoader,
        epochs: int,
    ) -> list:
        """Main trainer for net.
        Args:
            train_set: Training dataset of images.
            validate_set: Validation dataset.
            epochs: number of iterations to train
        Returns:
            self.valid_loss: Loss results from validation
        """
        for ep in range(epochs):
            self.train(train_set)
            self.validate(validate_set, ep)
            # Update learning rate
            self.lr_schedule.step()

        # Run final validation, and save model
        self.validate(validate_set, ep, save_model=True)
        self.run_iter += 1
        return self.valid_loss

    __call__ = run_solver

    def train(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Single training run, does one pass through the dataloader
        Args:
            dataloader: torch.utils.data.DataLoader

        Returns:
            None
        """

        len_batch = len(dataloader)
        for i_batch, sample_batched in enumerate(dataloader):
            print(f"Batch: [{i_batch}/{len_batch}]")

            rgb_img, depth_img, gt = sample_batched
            rgb_img = rgb_img.to(device)
            depth_img = depth_img.to(device)
            gt = gt.to(device)

            # Analagous to .zero_grad(), but faster
            for param in self.model.parameters():
                param.grad = None

            # Get predictions and restored depth map
            preds, restored_dep = self.model(rgb_img, depth_img)

            # Salient loss for predictions, BCE(pred) + Dice(pred)
            sal_loss = self.loss_fn(preds[0], gt) + self.dice_loss(preds[0], gt)
            for j in range(1, len(preds)):
                sal_loss += self.loss_fn(preds[j], gt) + self.dice_loss(preds[j], gt)

            # Restored Depth map loss
            idr_loss = 1 - self.idr_loss_fn(restored_dep, gt)

            # Total loss and optimize
            loss = sal_loss + self.lam * idr_loss
            loss.backward()
            self.optim.step()

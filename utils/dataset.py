import torch
import glob
import numpy as np
import imageio
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
import random

"""
Custom pytorch Dataset ref:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

Great SOD Dataset list:
    https://github.com/taozh2017/RGBD-SODsurvey
    http://dpfan.net/d3netbenchmark/
"""



def get_paths(rgb_dir, depth_dir, ground_truth_dir):
    # Get all image paths
    rgb_paths = [path for path in glob.glob(rgb_dir + "*.jpg")]
    depth_paths = [path for path in glob.glob(depth_dir + "*.png")]
    gt_paths = [path for path in glob.glob(ground_truth_dir + "*.png")]

    return rgb_paths, depth_paths, gt_paths



class NLPR(torch.utils.data.Dataset):
    """Only uses left RGB camera image"""

    def __init__(self, rgb_dir, depth_dir, ground_truth_dir):

        rgb_paths, depth_paths, gt_paths = get_paths(rgb_dir, depth_dir, ground_truth_dir)

        self.drgb_img = {}
        self.ddepth_map = {}
        self.dgt = {}

        self.transform = tf.Resize((320, 320))
        self.rgb_img = []
        self.depth_map = []
        self.gt = []

        # Get all unique image numbers
        for i in range(len(rgb_paths)):
            rgb = rgb_paths[i].replace(rgb_dir, '').replace('.jpg', '').replace('-', '').replace('_', '')
            depth = depth_paths[i].replace(depth_dir, '').replace('.png', '').replace('-', '').replace('_', '')
            gt = gt_paths[i].replace(ground_truth_dir, '').replace('.png', '').replace('-', '').replace('_', '')


            self.drgb_img[int(rgb)] = np.array(imageio.imread(rgb_paths[i]))
            self.ddepth_map[int(depth)] = np.array(imageio.imread(depth_paths[i]))
            self.dgt[int(gt)] = np.array(imageio.imread(gt_paths[i]))

        size = 0
        self.key_vals = {}
        for key in self.drgb_img.keys():
            if (key in self.ddepth_map) and (key in self.dgt):
                self.key_vals[size] = key
                size += 1

        self.leng = size

    def __len__(self):
        return self.leng

    def __getitem__(self, index):
        idx = self.key_vals[index]
        h, w, c = self.drgb_img[index].shape
        rgb = torch.Tensor(self.drgb_img[idx]).view(c, h, w)
        dep = torch.Tensor(
            self.ddepth_map[idx].reshape(
                1, self.ddepth_map[idx].shape[0], self.ddepth_map[idx].shape[1]
            )
        )
        gt = torch.Tensor(
            self.dgt[idx].reshape(1, self.dgt[idx].shape[0], self.dgt[idx].shape[1])
        )

        return (self.transform(rgb), self.transform(dep), self.transform(gt))


class NJU2K_Dataset(torch.utils.data.Dataset):
    """Only uses left RGB camera image"""

    def __init__(self, rgb_dir, depth_dir, ground_truth_dir):

        rgb_paths, depth_paths, gt_paths = get_paths(rgb_dir, depth_dir, ground_truth_dir)

        self.drgb_img = {}
        self.ddepth_map = {}
        self.dgt = {}

        self.transform = tf.Resize((320, 320))
        self.rgb_img = []
        self.depth_map = []
        self.gt = []

        # yapf: disable
        for i in range(len(rgb_paths)):
            rgb = rgb_paths[i].replace(rgb_dir, '').replace('_left.jpg', '')
            depth = depth_paths[i].replace(depth_dir, '').replace('_left.png', '')
            gt = gt_paths[i].replace(ground_truth_dir, '').replace('_left.png', '')

            self.drgb_img[int(rgb)] = np.moveaxis(np.array(imageio.imread(rgb_paths[i])), -1, 0)
            self.ddepth_map[int(depth)] = np.array(imageio.imread(depth_paths[i]))
            self.dgt[int(gt)] = np.array(imageio.imread(gt_paths[i]))

        size = 0
        self.key_vals = {}
        for key in self.drgb_img.keys():
            if (key in self.ddepth_map) and (key in self.dgt):
                self.key_vals[size] = key
                size += 1

        self.leng = size

    def __len__(self):
        return self.leng

    def __getitem__(self, index):
        idx = self.key_vals[index]
        rgb = torch.Tensor(self.drgb_img[idx])
        dep = torch.Tensor(
            self.ddepth_map[idx].reshape(
                1, self.ddepth_map[idx].shape[0], self.ddepth_map[idx].shape[1]
            )
        )
        gt = torch.Tensor(
            self.dgt[idx].reshape(1, self.dgt[idx].shape[0], self.dgt[idx].shape[1])
        )

        return (self.transform(rgb), self.transform(dep), self.transform(gt))




class DUT_Dataset(torch.utils.data.Dataset):
    """Only uses left RGB camera image"""

    def __init__(self, rgb_dir, depth_dir, ground_truth_dir):

        rgb_paths, depth_paths, gt_paths = get_paths(rgb_dir, depth_dir, ground_truth_dir)

        self.drgb_img = {}
        self.ddepth_map = {}
        self.dgt = {}

        self.transform = tf.Resize((320, 320))
        self.rgb_img = []
        self.depth_map = []
        self.gt = []

        # yapf: disable
        for i in range(len(rgb_paths)):
            rgb = rgb_paths[i].replace(rgb_dir, '').replace('.jpg', '')
            depth = depth_paths[i].replace(depth_dir, '').replace('.png', '')
            gt = gt_paths[i].replace(ground_truth_dir, '').replace('.png', '')

            self.drgb_img[int(rgb)] = np.moveaxis(np.array(imageio.imread(rgb_paths[i])), -1, 0)
            self.ddepth_map[int(depth)] = np.array(imageio.imread(depth_paths[i]))
            self.dgt[int(gt)] = np.array(imageio.imread(gt_paths[i]))

        size = 0
        self.key_vals = {}
        for key in self.drgb_img.keys():
            if (key in self.ddepth_map) and (key in self.dgt):
                self.key_vals[size] = key
                size += 1

        self.leng = size

    def __len__(self):
        return self.leng

    def __getitem__(self, index):
        idx = self.key_vals[index]
        rgb = torch.Tensor(self.drgb_img[idx])
        dep = torch.Tensor(
            self.ddepth_map[idx].reshape(
                1, self.ddepth_map[idx].shape[0], self.ddepth_map[idx].shape[1]
            )
        )
        gt = torch.Tensor(
            self.dgt[idx].reshape(1, self.dgt[idx].shape[0], self.dgt[idx].shape[1])
        )

        return (self.transform(rgb), self.transform(dep), self.transform(gt))

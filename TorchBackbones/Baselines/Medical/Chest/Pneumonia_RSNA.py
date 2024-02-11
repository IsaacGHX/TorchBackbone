import os
import time

import skimage
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import PIL

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as torchDataset
import torchvision as tv
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

import shutil

import pydicom

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.catch_warnings()

gpu_available = True

original_image_shape = 1024

datapath_orig = '../input/rsna-pneumonia-detection-challenge/'
datapath_prep = '../input/start-here-beginner-intro-to-lung-opacity-s1/'
datapath_out = '../input/pytorchunetpneumoniaoutput/'

# read train dataset
df_train = pd.read_csv(datapath_prep+'train.csv')
# read test dataset
df_test = pd.read_csv(datapath_prep+'test.csv')
df_train.head(3)
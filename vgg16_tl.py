from __future__ import print_function, division

import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

from torch.utils.data.sampler import SubsetRandomSampler

from utils import Load_Images

from PIL import Image

seed = 2000

np.random.seed(2000)
torch.manual_seed(200)

root_directory = "data/cars_train/"
carAnnotations_path = "data/devkit/cars_train_annos.mat"
metaData_path = "data/devkit/cars_meta.mat"

dataset = Load_Images(root_dir = root_directory, annotations_path=carAnnotations_path, seed=seed)

print(dataset)

'''
    class_names = loadmat(metadata_path)
    class_names = np.concatenate(class_names["class_names"][0])

    nb_classes = len(class_names)
'''
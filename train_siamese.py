from model import SiameseNetwork
from dataset import SiameseNetworkDataset
from loss import ContrastiveLoss
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms as T
import os
import torchvision
import matplotlib.pyplot as plt

LEARNING_RATE = 0.0001
BATCH_SIZE = 10
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available else "cpu"

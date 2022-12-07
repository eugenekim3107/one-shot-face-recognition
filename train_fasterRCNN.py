import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm
from torchvision.utils import draw_bounding_boxes
from os.path import abspath, expanduser
from typing import Dict, List, Union
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataset import WIDERFace, Compose2

#

model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
n_classes = 1
in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, nesterov=True, weight_decay=1e-4)
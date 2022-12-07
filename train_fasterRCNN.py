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
import vessl

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 30
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
EPOCHS = 50
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "model1.pth.tar"
IMG_DIR = "images"
LABEL_DIR = "labels"
DIR_NAME = "output"

model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
n_classes = 1
in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True, weight_decay=WEIGHT_DECAY)

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()
    all_losses = []
    all_losses_dict = []
    all_mAP = []
    for images, targets in tqdm(loader):
        images = list(image.to(device) for image in images)
        targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t
                   in targets]

        loss_dict = model(images,
                          targets)  # the model computes the loss automatically if we pass in targets
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)

        model.eval()
        preds = model(images)
        metric = MeanAveragePrecision()
        metric.update(preds, targets)
        map = metric.compute()["map"]
        all_mAP.append(map)
        model.train()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    all_losses_dict = pd.DataFrame(all_losses_dict)
    print(
        "Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}, mAP: {:.6f}".format(
            epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
            all_losses_dict['loss_box_reg'].mean(),
            all_losses_dict['loss_rpn_box_reg'].mean(),
            all_losses_dict['loss_objectness'].mean(),
            map
        ))



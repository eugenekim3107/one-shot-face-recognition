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
from tqdm import tqdm
from torchvision.utils import draw_bounding_boxes
from os.path import abspath, expanduser
from typing import Dict, List, Union
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataset import faceDatasetFasterRCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_one_epoch(model, optimizer, loader, device, epoch):
    model.to(device)
    model.train()

    all_losses = []
    all_losses_dict = []

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

        if not math.isfinite(loss_value):
            print(
                f"Loss is {loss_value}, stopping trainig")  # train if loss becomes infinity
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing
    print(
        "Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
            epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
            all_losses_dict['loss_classifier'].mean(),
            all_losses_dict['loss_box_reg'].mean(),
            all_losses_dict['loss_rpn_box_reg'].mean(),
            all_losses_dict['loss_objectness'].mean()
        ))


def test_one_epoch(model, loader, device, epoch):
    model.eval()
    model.to(device)
    metric = MeanAveragePrecision(box_format="xywh", iou_type="bbox")
    metric.to(device)

    all_mAP = []

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = list(image.to(device) for image in images)
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for
                       t in targets]

            predictions = model(images)
            metric.update(predictions, targets)
            single_mAP = metric.compute()
            all_mAP.append(single_mAP)

    all_mAP = pd.DataFrame(all_mAP)
    print("Epoch {}, mAP: {:.6f}".format(
        epoch, all_mAP["map"].mean()
    ))

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(600, 600), # our input size can be 600px
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform
def collate_fn(batch):
    return tuple(zip(*batch))
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    data = faceDatasetFasterRCNN("faceYoloData.csv", "images", "labels", "data",
                           transforms=get_transforms(True))
    train_data, test_data = torch.utils.data.random_split(data, [
        int(0.8 * len(data)) + 1, int(0.2 * len(data))])
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False,
                             collate_fn=collate_fn)
    torch.cuda.empty_cache()
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # we need to change the head
    n_classes = 3
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, n_classes)
    device = torch.device("cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True,
                                weight_decay=1e-4)

    num_epochs = 20
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        test_one_epoch(model, test_loader, device, epoch)


if __name__ == "__main__":
    main()


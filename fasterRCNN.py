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
import albumentations as A  # our data augmentation library
import os
from os.path import abspath, expanduser
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
from albumentations.pytorch import ToTensorV2
import sys

DIR_NAME = "output"
LOAD_MODEL_FILE = "modelrcnn.pth.tar"

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


class WIDERFace(Dataset):
    BASE_FOLDER = "widerface"

    def __init__(self, root, split="train", transform=None):
        # check arguments
        self.root = os.path.join(root, self.BASE_FOLDER)
        self.transform = transform
        self.split = split
        self.img_info: List[Dict[str, Union[str, Dict[str, torch.Tensor]]]] = []
        if self.split in ("train", "val"):
            self.parse_train_val_annotations_file()
        else:
            self.parse_test_annotations_file()

    def __getitem__(self, index: int):
        # image = Image.open(self.img_info[index]["img_path"]).convert("RGB")
        image = cv2.imread(self.img_info[index]["img_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = self.img_info[index]["annotations"]["bbox"]

        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=boxes)

        image = transformed['image']
        boxes = transformed['bboxes']

        new_boxes = []  # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(new_boxes, dtype=torch.float32)

        targ = {}  # here is our transformed target
        targ['boxes'] = boxes
        targ['labels'] = torch.tensor([1 for x in boxes], dtype=torch.int64)
        targ['image_id'] = torch.tensor([1 for x in boxes], dtype=torch.int64)
        targ['area'] = (boxes[:, 3] - boxes[:, 1]) * (
                    boxes[:, 2] - boxes[:, 0])  # we have a different area
        targ['iscrowd'] = torch.tensor([1 for x in boxes], dtype=torch.int64)

        return image.div(255), targ  # scale images

    def __len__(self):
        return len(self.img_info)

    def extra_repr(self):
        lines = ["Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)

    def parse_train_val_annotations_file(self):
        filename = "wider_face_train_bbx_gt.txt" if self.split == "train" else "wider_face_val_bbx_gt.txt"
        filepath = os.path.join(self.root, "wider_face_split", filename)

        with open(filepath) as f:
            lines = f.readlines()
            file_name_line, num_boxes_line, box_annotation_line = True, False, False
            num_boxes, box_counter = 0, 0
            labels = []
            for line in lines:
                line = line.rstrip()
                if file_name_line:
                    img_path = os.path.join(self.root, "WIDER_" + self.split,
                                            "images", line)
                    if not os.path.isfile(img_path):
                        continue
                    img_path = abspath(expanduser(img_path))
                    file_name_line = False
                    num_boxes_line = True
                elif num_boxes_line:
                    num_boxes = int(line)
                    num_boxes_line = False
                    box_annotation_line = True
                elif box_annotation_line:
                    box_counter += 1
                    line_split = line.split(" ")
                    line_values = [int(x) for x in line_split]
                    labels.append(line_values)
                    if box_counter >= num_boxes:
                        box_annotation_line = False
                        file_name_line = True
                        labels_tensor = torch.tensor(labels)[:, 0:4]
                        self.img_info.append(
                            {
                                "img_path": img_path,
                                "annotations": {
                                    "bbox": torch.cat((labels_tensor,
                                                       torch.ones(
                                                           labels_tensor.shape[
                                                               0], 1)), axis=1)
                                    # x, y, width, height
                                },
                            }
                        )
                        box_counter = 0
                        labels.clear()
                else:
                    raise RuntimeError(
                        f"Error parsing annotation file {filepath}")

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

    #         if lr_scheduler is not None:
    #             lr_scheduler.step() #

    all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing
    print(
        "Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
            epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
            all_losses_dict['loss_classifier'].mean(),
            all_losses_dict['loss_box_reg'].mean(),
            all_losses_dict['loss_rpn_box_reg'].mean(),
            all_losses_dict['loss_objectness'].mean()
        ))

def main():
    train_dataset = WIDERFace("data/data", "train", get_transforms(True))
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    device = torch.device("cuda")
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
    num_epochs=10

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "start_epoch": num_epochs
    }
    save_checkpoint(DIR_NAME, checkpoint, filename=LOAD_MODEL_FILE)

if __name__ == "__main__":
    main()
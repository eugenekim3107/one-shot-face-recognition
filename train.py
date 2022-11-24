"""
Main file for training Yolo model on Pascal VOC dataset
"""

import torch
import time
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
import vessl
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import faceYoloDataset, WIDERFace
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
import os

vessl.init()

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 300
WEIGHT_DECAY = 0
EPOCHS = 50
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "model1.pth.tar"
IMG_DIR = "images"
LABEL_DIR = "labels"
DIR_NAME = "output"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn, epoch, start_epoch):
    model.train()
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    vessl.log(
        step=epoch + start_epoch + 1,
        payload={'train_loss': sum(mean_loss)/len(mean_loss)}
    )

    print(f"Mean training loss was {sum(mean_loss)/len(mean_loss)}")

def test_fn(test_loader, model, loss_fn, epoch, start_epoch):
    model.eval()
    loop = tqdm(test_loader, leave=True)
    mean_test_loss = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            mean_test_loss.append(loss.item())

            loop.set_postfix(loss=loss.item())

    vessl.log(
        step=epoch + start_epoch + 1,
        payload={'test_loss': sum(mean_test_loss)/len(mean_test_loss)}
    )

    print(f"Mean test loss was {sum(mean_test_loss)/len(mean_test_loss)}")


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=2).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        checkpoint = torch.load(LOAD_MODEL_FILE)
        start_epoch = load_checkpoint(checkpoint, model, optimizer)
    else:
        start_epoch = 0

    # dataset = faceYoloDataset(
    #     "faceYoloData.csv",
    #     transform=transform,
    #     img_dir=IMG_DIR,
    #     label_dir=LABEL_DIR,
    #     data_dir="data/data"
    # )
    train_dataset = WIDERFace(root="data/data/", split="train", transform=transform)
    test_dataset = WIDERFace(root="data/data/", split="val", transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    epoch = 0
    for epoch in range(EPOCHS):
        # for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(8):
        #        bboxes = cellboxes_to_boxes(y)
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
        #
        #    import sys
        #    sys.exit()

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        pred_boxes_test, target_boxes_test = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

        mean_avg_prec_test = mean_average_precision(
            pred_boxes_test, target_boxes_test, iou_threshold=0.5, box_format="midpoint"
        )

        print(f"[{epoch+1}] Train mAP: {mean_avg_prec}")
        print(f"[{epoch+1}] Test mAP: {mean_avg_prec_test}")
        vessl.log(
            step=epoch + start_epoch + 1,
            payload={"train_mAP": float(mean_avg_prec)}
        )
        vessl.log(
            step=epoch + start_epoch + 1,
            payload={"test_mAP": float(mean_avg_prec_test)}
        )

        if mean_avg_prec > 0.9:
           checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
               "start_epoch": epoch
           }
           save_checkpoint(DIR_NAME, checkpoint, filename=LOAD_MODEL_FILE)
           time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn, epoch, start_epoch)
        test_fn(test_loader, model, loss_fn, epoch, start_epoch)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "start_epoch": epoch
    }
    save_checkpoint(DIR_NAME, checkpoint, filename=LOAD_MODEL_FILE)
    time.sleep(10)

if __name__ == "__main__":
    main()
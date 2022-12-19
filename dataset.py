import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import pandas as pd
from torchvision import transforms as T
from PIL import Image
from utils import (plot_image, cellboxes_to_boxes, non_max_suppression)
from os.path import abspath, expanduser
from typing import Dict, List, Union
import numpy as np
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

class faceYoloDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, data_dir, S=7, B=2, C=2, transform=None):
        self.annotations = pd.read_csv(os.path.join(data_dir,csv_file))
        self.img_dir = os.path.join(data_dir, img_dir)
        self.label_dir = os.path.join(data_dir, label_dir)
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index,1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:

            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)

            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1
        return image, label_matrix


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
        img = Image.open(self.img_info[index]["img_path"]).convert("RGB")
        boxes = self.img_info[index]["annotations"]["bbox"]

        img_shape = np.array(img)
        img_h = img_shape.shape[0]
        img_w = img_shape.shape[1]
        boxes[:, 0] /= img_w
        boxes[:, 1] /= img_h
        boxes[:, 2] /= img_w
        boxes[:, 3] /= img_h

        if self.transform:
            img, boxes = self.transform(img, boxes)

        new_boxes = []  # convert from xywh to xyxy
        for box in boxes:
            xmin = box[0]
            xmax = xmin + box[2]
            ymin = box[1]
            ymax = ymin + box[3]
            new_boxes.append([xmin, ymin, xmax, ymax])

        new_boxes = torch.tensor(new_boxes, dtype=torch.float32)

        label = {"boxes": new_boxes,
                 "labels": torch.tensor([0 for i in range(new_boxes.size(0))],
                                       dtype=torch.int64)}
        return img, label

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
                        labels_tensor = torch.tensor(labels)[:, 0:4].float()
                        self.img_info.append(
                            {
                                "img_path": img_path,
                                "annotations": {
                                    "bbox": labels_tensor
                                    # x, y, width, height
                                },
                            }
                        )
                        box_counter = 0
                        labels.clear()
                else:
                    raise RuntimeError(
                        f"Error parsing annotation file {filepath}")


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = self.imageFolderDataset.imgs[
            np.random.choice(len(self.imageFolderDataset.imgs))]
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = np.random.randint(0, 2)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = self.imageFolderDataset.imgs[
                    np.random.choice(len(self.imageFolderDataset.imgs))]
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = self.imageFolderDataset.imgs[
                    np.random.choice(len(self.imageFolderDataset.imgs))]
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(
            np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

class Compose2(object):
    def __init__(self):
        self.transforms = [T.Resize(size=(400, 400)), T.ToTensor()]

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img = t(img)

        return img, bboxes



def main():

    ### For YOLO Dataset ###

    # csv_file = "faceYoloData.csv"
    # img_dir = "images"
    # label_dir = "labels"
    # data_dir = "data"
    # batch_size = 1
    #
    # transform = Compose([T.Resize((448, 448)), T.ToTensor()])
    # dataset = faceYoloDataset(
    #     csv_file,
    #     transform=transform,
    #     img_dir=img_dir,
    #     label_dir=label_dir,
    #     data_dir=data_dir
    # )
    # print(len(dataset))
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # for x, y in train_loader:
    #     for idx in range(batch_size):
    #         bboxes = cellboxes_to_boxes(y)
    #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5,
    #                                      threshold=0.4, box_format="midpoint")
    #         plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes)
    #     break

    ### WIDER FACE Dataset ###

    transform = Compose2()
    train_data = WIDERFace(root="data/", split="train", transform=transform)
    test_data = WIDERFace(root="data/", split="val", transform=transform)

    def collate_fn(batch):
        return tuple(zip(*batch))

    batch_size = 1
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn)

    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    images, targets = next(iter(train_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    images_test, targets_test = next(iter(test_loader))
    images_test = list(image for image in images_test)
    targets_test = [{k: v for k, v in t.items()} for t in targets_test]
    print(targets_test[0])

    for i, (x, y) in enumerate(test_loader):
        boxes = y[0]["boxes"]
        show(draw_bounding_boxes((x[0]*255).type(torch.uint8), (boxes*400)))
        break


if __name__ == "__main__":
    main()






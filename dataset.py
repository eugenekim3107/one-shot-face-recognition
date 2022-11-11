import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
from PIL import Image
from utils import (plot_image, cellboxes_to_boxes, non_max_suppression)

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
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
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

def main():
    csv_file = "faceYoloData.csv"
    img_dir = "images"
    label_dir = "labels"
    data_dir = "data"
    batch_size = 1

    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, img, bboxes):
            for t in self.transforms:
                img, bboxes = t(img), bboxes

            return img, bboxes

    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

    data = faceYoloDataset(csv_file=csv_file,
                           img_dir=img_dir,
                           label_dir=label_dir,
                           data_dir=data_dir,
                           transform=transform
                           )

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    for (image, label) in train_loader:
        for idx in range(8):
            bboxes = cellboxes_to_boxes(label)
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            plot_image(image[idx].permute(1,2,0), bboxes)
        break

if __name__ == "__main__":
    main()






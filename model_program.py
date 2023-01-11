import argparse
import os
import torch
import matplotlib.pyplot as plt
from model import SiameseNetwork, resnet34
from torchvision import models
from torch.autograd import Variable
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image, ImageFont
from torchvision import transforms
from torch.nn import functional as F
from torchvision.utils import draw_bounding_boxes
import torchvision

parser = argparse.ArgumentParser()
parser.add_argument(
    '--group-image',
    default='group_image/example1.png',
    help='Group Image for Support Set')
parser.add_argument(
    '--query-image',
    default="query_image/query3.jpg",
    help='Query Image of Individual'
)
parser.add_argument(
    '--name',
    default='Unknown',
    help='Name of Query Individual'
)

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(800, 800),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(800, 800),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform

def fasterRCNN(model, group_image):

    model.eval()
    with torch.no_grad():
        prediction = model([group_image])
        pred = prediction[0]
    return pred

def siamese(model, query_image, support_set):

    model.eval()
    transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    query_image = transform(query_image.convert("L")).unsqueeze(0)
    lowest_score = np.inf
    lowest_idx = None

    for i, img in enumerate(support_set):
        temp_img = transform(img.convert("L")).unsqueeze(0)
        output1, output2 = model(Variable(query_image), Variable(temp_img))
        euclidean_distance = F.pairwise_distance(output1, output2)

        if euclidean_distance.item() < lowest_score:
            lowest_score = euclidean_distance
            lowest_idx = i

    return lowest_score, lowest_idx

def get_faces(image, boxes):
    im = image.permute(1,2,0)
    height, width, _ = im.shape
    list_img = []
    transform = transforms.ToPILImage()

    for i, box in enumerate(boxes):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        img = im[y1:y2, x1:x2, :]
        img = transform(img.permute(2,0,1))
        list_img.append(img)

    return list_img

def main(args):

    group_image = Image.open(args.group_image).convert("RGB")
    query_image = Image.open(args.query_image).convert("RGB")
    h,w,c = np.array(group_image).shape
    ratio = h/w

    # Create fasterRCNN model and siamese model
    fasterRCNN_model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights="FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT", pretrained=True)
    in_features = fasterRCNN_model.roi_heads.box_predictor.cls_score.in_features
    n_classes = 3
    fasterRCNN_model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, n_classes)
    siamese_model = SiameseNetwork()
    # resnet34_model = resnet34()

    # Load state for both models
    checkpoint = torch.load("load_models/fasterRCNNstate.pth.tar",
                            map_location=torch.device("cpu"))
    fasterRCNN_model.load_state_dict(checkpoint)
    checkpoint = torch.load("load_models/siamesestate.pth.tar",
                            map_location=torch.device("cpu"))
    # resnet34_model.load_state_dict(checkpoint)
    siamese_model.load_state_dict(checkpoint)

    # transform group_image
    transform = get_transforms(train=False)
    transformed = transform(image=np.array(group_image), bboxes=[])
    group_image = (transformed["image"]).div(255)

    # Retrieve bounding boxes from face localization
    pred = fasterRCNN(fasterRCNN_model, group_image)
    threshold = 0.8
    boxes = pred["boxes"][pred["scores"] > threshold]
    support_set = get_faces(group_image, boxes)

    # Evaluate dissimilarity score for each image in support set and find best match
    lowest_score, lowest_idx = siamese(model=siamese_model, query_image=query_image, support_set=support_set)
    # lowest_score, lowest_idx = siamese(model=resnet34_model,
    #                                    query_image=query_image,
    #                                    support_set=support_set)

    # Return the image with best bounding box
    final = torch.unsqueeze(pred["boxes"][lowest_idx], 0)
    group_image = group_image.clone().detach() * 255
    group_image = group_image.to(torch.uint8)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(15, int(15*ratio))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(draw_bounding_boxes(group_image, final, labels=[args.name],
                                   font='~/Library/Fonts/Arial.ttf',
                                   font_size=20,
                                   width=3,
                                   colors="red").permute(1, 2, 0), aspect='auto')
    plt.savefig("test.jpg")

if __name__ == '__main__':
    main(parser.parse_args())
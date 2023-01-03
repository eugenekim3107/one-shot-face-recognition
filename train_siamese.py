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
from model import SiameseNetwork
from dataset import SiameseNetworkDataset
from loss import ContrastiveLoss

batch_size = 10
epochs = 10
threshold = 0.3
lr = 0.0005

transform=T.Compose([T.Resize((100,100)),T.ToTensor()])
train = torchvision.datasets.ImageFolder(root='data/celebDataset/train')
test = torchvision.datasets.ImageFolder(root='data/celebDataset/test')
data = SiameseNetworkDataset(train,transform=transform,should_invert=False)
testdata = SiameseNetworkDataset(test,transform=transform,should_invert=False)
def collate_fn(batch):
    img1 = []
    img2 = []
    labels = []
    for i in batch:
        img1.append(i[0])
        img2.append(i[1])
        labels.append(i[2])
    return torch.stack(img1), torch.stack(img2), torch.stack(labels)

train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(testdata, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

loss_history = []
accuracy_history = []
test_accuracy = []

for epoch in range(epochs):
    net.train()
    losses = []
    accuracies = []

    for i, data in enumerate(train_loader):
        img0, img1, label = data
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive.backward()
        optimizer.step()

        accuracy = sum((euclidean_distance > threshold) == label.squeeze()) / batch_size
        losses.append(loss_contrastive.item())
        accuracies.append(accuracy)
    loss_history.append(np.mean(losses))
    accuracy_history.append(np.mean(accuracies))
    print(
        f"Epoch {epoch + 1}: Train Loss = {np.mean(losses)}, Train Accuracy = {np.mean(accuracies)}")

    net.eval()
    accuracies = []

    for i, data in enumerate(test_loader):
        img0, img1, label = data
        output1, output2 = net(img0, img1)
        euclidean_distance = F.pairwise_distance(output1, output2)
        accuracy = sum((euclidean_distance > threshold) == label.squeeze()) / batch_size
        accuracies.append(accuracy)
    test_accuracy.append(np.mean(accuracies))
    print(
        f"Epoch {epoch + 1}: Test Accuracy = {np.mean(accuracies)}")



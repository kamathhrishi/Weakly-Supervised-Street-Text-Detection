import os
import torch
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
import yaml
import torch
import matplotlib.pyplot as plt
import random
import cv2
import math
import torch.optim as optim
from utils import plot_gallery
from train_utils import save_model
from utils import get_mask
from dataset import shuffle_loader, CharsDataset


class Model(nn.Module):
    def __init__(self):

        super(Model, self).__init__()
        net = models.alexnet(pretrained=True)
        self.features = net.features[0:10]
        self.adaptivepool = nn.AdaptiveAvgPool2d(output_size=(50, 50))
        self.conv1 = nn.Conv2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.classifier = nn.Conv2d(
            256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.adaptivepool(x)
        x = self.conv1(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)

        return x


def load_dataset(args, traindata_transform, testdata_transform):

    Images = []
    labels = []

    for i in os.listdir("new_results/images"):

        if (
            i != "annotations"
            and i != ".DS_Store"
            and i != "res"
            and i != "images"
            and i != "results"
        ):

            img = Image.open("new_results/images/" + i)
            # display(img)
            Images.append(img.resize((250, 250)))
            fp = open("new_results/annotations/" + i[:-4] + ".yaml", "r")
            mask = get_mask(yaml.load(fp))
            # plt.imshow(mask, cmap='hot', interpolation='nearest')
            # plt.show()
            labels.append(mask)

    X_train, X_test, y_train, y_test = train_test_split(
        Images, labels, test_size=0.3, random_state=42
    )

    traindataset = CharsDataset(X_train, y_train, transform=traindata_transform)
    trainloader = shuffle_loader(traindataset, args.batchsize_trainloader)

    testdataset = CharsDataset(X_test, y_test, transform=testdata_transform)
    testloader = shuffle_loader(testdataset, args.batchsize_testloader)

    return trainloader, testloader


def evaluate(model, test_loader):

    running_loss = 0.0
    index = 0

    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        criterion = nn.BCELoss()
        outputs = model(inputs.double())
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        index += 1

    return running_loss / index


def train(model, trainloader, testloader, criterion, optimizer, testtransform):

    best_error = 99999999999999.0

    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        index = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs.double())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            index += 1
            print(
                "Epoch ",
                epoch,
                " (Index ",
                str(index),
                "/",
                str(len(trainloader)),
                " Loss : ",
                loss.item(),
                ")",
            )
        test_error = evaluate(model, testloader)
        if test_error < best_error:

            best_error = test_error
            save_model(model, optimizer, name="models/localization_model.pth")
        print("Running Error: ", running_loss / len(trainloader))
        print("Test Error: ", test_error)
        print("Best Error: ", best_error)
        # plot_gallery(model, testtransform)


class Arguments:
    def __init__(self):

        self.random_seed = 1
        self.batch_size = 16
        self.lr = 0.001
        self.momentum = 0.9
        self.batchsize_trainloader = 16
        self.batchsize_testloader = 1000


def main():

    traindata_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    testdata_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    args = Arguments()

    trainloader, testloader = load_dataset(
        args, traindata_transform, testdata_transform
    )

    model = Model().double()

    torch.manual_seed(args.random_seed)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    train(model, trainloader, testloader, criterion, optimizer, testdata_transform)


main()

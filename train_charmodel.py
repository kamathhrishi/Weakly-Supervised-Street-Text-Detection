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
import torch
import random
import torch.optim as optim
from train_utils import evaluate, save_model
from utils import crop_img
from dataset import prepare_dataset


def load_data(args):

    images = []
    labels = []
    index = 0

    path = "Chars74k/English/Img/GoodImg/"

    for i in os.listdir(path + "Bmp"):

        if i != ".DS_Store":

            for j in os.listdir(path + "Bmp" + "/" + i):

                if j != ".DS_Store":

                    img = (
                        Image.open(path + "Bmp" + "/" + i + "/" + j)
                        .convert("RGB")
                        .resize((args.IMG_SIZE, args.IMG_SIZE))
                    )
                    images.append(img)
                    labels.append(1)

            index += 1

    for i in os.listdir("Background"):

        if i != ".DS_Store":

            img = Image.open("Background" + "/" + i).convert("RGB")

            cropped_imgs = crop_img(img, args.IMG_SIZE)

            for i in cropped_imgs:

                images.append(i.resize((args.IMG_SIZE, args.IMG_SIZE)))
                labels.append(0)

    return images, labels


def train(model, trainloader, testloader, criterion, optimizer):

    best_accuracy = 0.0

    for epoch in range(20):  # loop over the dataset multiple times

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

        test_accuracy = evaluate(model.double(), testloader, "Test Accuracy")

        if test_accuracy >= best_accuracy:

            save_model(model, optimizer, name="models/char_model.pth")

        print("loss: ", running_loss / len(trainloader))

    print("Finished Training")


class Arguments:
    def __init__(self):

        self.random_seed = 1
        self.batch_size = 16
        self.lr = 0.001
        self.momentum = 0.9
        self.IMG_SIZE = 120
        self.traindata_batchsize = 16
        self.testdata_batchsize = 1000


def main():

    traindata_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
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

    net = models.alexnet(pretrained=True)
    net.classifier[6] = nn.Linear(4096, 2)
    net = net.double()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    torch.manual_seed(args.random_seed)

    images, labels = load_data(args)
    trainloader, testloader = prepare_dataset(
        images, labels, args, traindata_transform, testdata_transform
    )
    train(net, trainloader, testloader, criterion, optimizer)


main()

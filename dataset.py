import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import os
from PIL import Image
from utils import crop_img
import torch


class CharsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, chars, labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
        self.chars = chars
        self.labels = labels

    def __len__(self):

        return len(self.chars)

    def __getitem__(self, idx):

        # print(self.scenes[idx].shape)
        return self.transform(self.chars[idx]), self.labels[idx]


def shuffle_loader(data, batch_size, shuffle_dataset=True, random_seed=42):

    dataset_size = len(data)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Creating PT data samplers and loaders:
    sampler = SubsetRandomSampler(indices)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=sampler)

    return loader


def prepare_dataset(images, labels, args, traindata_transform, testdata_transform):

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.33, random_state=42
    )

    traindataset = CharsDataset(X_train, y_train, transform=traindata_transform)
    trainloader = shuffle_loader(traindataset, args.traindata_batchsize)

    testdataset = CharsDataset(X_test, y_test, transform=testdata_transform)
    testloader = shuffle_loader(testdataset, args.testdata_batchsize)

    return trainloader, testloader

import os
import glob

import torch
import torchvision
from torchvision import transforms

from .cifar_dataloader import CIFAR10Dataset

from plots import plot_sample_images

def load_data(DIR_TRAIN, DIR_TEST):

    classes = os.listdir(DIR_TRAIN)
    print("Total Classes: ", len(classes))
    print(classes)
    train_imgs = []
    test_imgs  = []
    for _class in classes:
        train_imgs += [os.path.normpath(i) for i in glob.glob(DIR_TRAIN + _class + '/*.png')]
        test_imgs += [os.path.normpath(i) for i in glob.glob(DIR_TEST + _class + '/*.png')]

    print("\nTotal train images: ", len(train_imgs))
    print("Total test images: ", len(test_imgs))

    cifar_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])

    train_dataset = CIFAR10Dataset(imgs_list = train_imgs, classes = classes, transforms = cifar_transforms)
    # plot_sample_images(train_dataset)

    test_dataset = CIFAR10Dataset(imgs_list = test_imgs, classes = classes, transforms = cifar_transforms)
    # plot_sample_images(test_dataset)

    return train_dataset, test_dataset


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform


class CIFAR10Dataset(Dataset):
    def __init__(self, imgs_list, classes, transforms=None):
        super(CIFAR10Dataset, self).__init__()
        self.imgs_list = imgs_list
        self.class_to_int = {classes[i] : i for i in range(len(classes))}
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.imgs_list[index]
        
        # Reading image
        image = io.imread(image_path) #Image.open(image_path)
        # Retriving class label
        label = image_path.split("\\")[-2]
        label = self.class_to_int[label]
        
        # Applying transforms on image
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label      

    def __len__(self):
        return len(self.imgs_list)
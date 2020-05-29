from torchvision import transforms
import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD)

class JASDataset(Dataset):
    def __init__(self, path_to_csv, images_path):
        self.csv_path = path_to_csv
        self.df = pd.read_csv(self.csv_path)
        self.images_path = images_path
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            normalize])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = row['Composition']##label
        y = y / 100.
        image_name = row['name']
        image_path = os.path.join(self.images_path, image_name)
        # image = default_loader(image_path)
        image = Image.open(image_path)
        if self.csv_path[34:] =='train.csv':
            x = self.train_transform(image)
        else:
            x = self.val_transform(image)
        return x, y.astype('float32')
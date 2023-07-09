import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset import TextDataset, ImageDataset, MultimodalDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class TextDataloader():
    def __init__(self, data, data_folder,pretrained_TextModel, max_len, batch_size, validation_size):
        self.data = data
        self.data_folder = data_folder
        self.pretrained_TextModel = pretrained_TextModel
        self.batch_size = batch_size
        self.max_len=max_len
        self.validation_size = validation_size

    def load_data(self):
        loaded_data = []
        with open(self.data, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines[1:]):
                parts = line.strip().split(',')
                guid = parts[0]
                tag = parts[1]
                text_name=os.path.join(self.data_folder, guid + '.txt')
                
                with open(text_name, 'rb') as text_file:
                    text = text_file.read().strip().decode("utf-8","ignore")

                loaded_data.append((text, tag))
        train_data, valid_data = train_test_split(loaded_data, test_size=self.validation_size, random_state=1423)
        return train_data, valid_data

    def __call__(self):
        train_data, valid_data = self.load_data()

        train_dataset = TextDataset(train_data, self.pretrained_TextModel, self.max_len)
        valid_dataset = TextDataset(valid_data, self.pretrained_TextModel, self.max_len)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, valid_dataloader


class ImageDataloader():
    def __init__(self, data, data_folder, transform, batch_size, validation_size):
        self.data = data
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.transform = transform
        self.validation_size = validation_size

    def load_data(self):
        loaded_data = []
        with open(self.data, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines[1:]):
                parts = line.strip().split(',')
                guid = parts[0]
                tag = parts[1]
                img_name = os.path.join(self.data_folder, guid + '.jpg')
            
                with Image.open(img_name) as img:
                    img = self.transform(img)

                loaded_data.append((img, tag))
        train_data, valid_data = train_test_split(loaded_data, test_size=self.validation_size, random_state=1423)
        return train_data, valid_data

    def __call__(self):
        train_data, valid_data = self.load_data()

        train_dataset = ImageDataset(train_data, self.pretrained_TextModel, self.max_len)
        valid_dataset = ImageDataset(valid_data, self.pretrained_TextModel, self.max_len)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, valid_dataloader


class MultimodalDataloader():
    def __init__(self, data, data_folder,pretrained_TextModel, transform,max_len, batch_size, validation_size):
        self.data = data
        self.data_folder = data_folder
        self.pretrained_TextModel = pretrained_TextModel
        self.batch_size = batch_size
        self.transform = transform
        self.max_len=max_len
        self.validation_size = validation_size

    def load_data(self):
        loaded_data = []
        with open(self.data, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines[1:]):
                parts = line.strip().split(',')
                guid = parts[0]
                tag = parts[1]
                text_name=os.path.join(self.data_folder, guid + '.txt')
                img_name = os.path.join(self.data_folder, guid + '.jpg')
                
                with open(text_name, 'rb') as text_file:
                    text = text_file.read().strip().decode("utf-8","ignore")

                with Image.open(img_name) as img:
                    img = self.transform(img)

                loaded_data.append((img, text, tag))
        train_data, valid_data = train_test_split(loaded_data, test_size=self.validation_size, random_state=1423)
        return train_data, valid_data

    def __call__(self):
        train_data, valid_data = self.load_data()

        train_dataset = MultimodalDataset(train_data, self.pretrained_TextModel, self.max_len)
        valid_dataset = MultimodalDataset(valid_data, self.pretrained_TextModel, self.max_len)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, valid_dataloader


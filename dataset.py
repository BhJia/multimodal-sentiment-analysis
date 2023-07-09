import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, data, pretrained_TextModel,max_len):
        self.data = data
        self.pretrained_TextModel = pretrained_TextModel
        self.max_len=max_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_TextModel)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)    
        label = torch.tensor(0 if label == 'negative' else 1 if label == 'neutral' else 2)
        return inputs, label
        

class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img,label = self.data[idx]    
        label = torch.tensor(0 if label == 'negative' else 1 if label == 'neutral' else 2)

        return img, label


class MultimodalDataset(Dataset):
    def __init__(self, data, pretrained_TextModel,max_len):
        self.data = data
        self.pretrained_TextModel = pretrained_TextModel
        self.max_len=max_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_TextModel)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, text, label = self.data[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)    
        label = torch.tensor(0 if label == 'negative' else 1 if label == 'neutral' else 2)

        return img, inputs, label


class MultimodalDatasetWithGuid(Dataset):
    def __init__(self, data, pretrained_TextModel,max_len):
        self.data = data
        self.pretrained_TextModel = pretrained_TextModel
        self.max_len=max_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_TextModel)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        guid, img, text, label = self.data[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)    
        label = torch.tensor(0 if label == 'negative' else 1 if label == 'neutral' else 2)

        return guid, img, inputs, label
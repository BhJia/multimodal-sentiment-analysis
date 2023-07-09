import os
import sys
import time
import torch
from torch import nn
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from transformers import logging as transformer_logging
transformer_logging.set_verbosity_error()
from PIL import Image
import numpy as np
import pandas as pd
from dataset import MultimodalDatasetWithGuid
from dataloader import MultimodalDataloader
from sklearn.metrics import accuracy_score
from model import MultimodalModel
from tqdm import tqdm
from utils import *
import wandb
import logging
import argparse

# set args
parser = argparse.ArgumentParser("Multimodal-Sentiment-Analysis")
parser.add_argument('--test_data', type=str, default="data/test_without_label.txt", help='path of input testing data')
parser.add_argument('--data_folder', type=str, default="data/data", help='path of input data folder')
parser.add_argument('--text_model', type=str, default="roberta", help='model used for texts')
parser.add_argument('--image_model', type=str, default="resnet", help='model used for images')
parser.add_argument('--pretrained_text', type=str, default="pretrained/sentiment-roberta",
                    help='path of pretrained text model')
parser.add_argument('--pretrained_image', type=str, default="pretrained/resnet/resnet50.pth",
                    help='path of pretrained image model')
parser.add_argument('--fusion_method', type=str, default="concat", help='the fusion method')
parser.add_argument('--weights', type=str, default='checkpoints/best_model.pth', 
                    help='training checkpoints')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--max_len', type=int, default=256, help='the maximun length of the sequence')
parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default="0", help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--save_path', type=str, default='results', help='path of the output results')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def test_data_loader(test_data):
    loaded_data = []
    with open(test_data, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines[1:]):
            parts = line.strip().split(',')
            guid = parts[0]
            tag = parts[1]
            text_name = os.path.join(args.data_folder, guid + '.txt')
            img_name = os.path.join(args.data_folder, guid + '.jpg')
            
            with open(text_name, 'rb') as text_file:
                text = text_file.read().strip().decode("utf-8","ignore")

            with Image.open(img_name) as img:
                img = transform(img)

            loaded_data.append((guid, img, text, tag))

    test_dataset = MultimodalDatasetWithGuid(loaded_data, args.pretrained_text, args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return test_dataloader


def predict(model, test_dataloader):
    print("start testing")
    model.eval()
    Guids=[]
    predictions=[]
    with torch.no_grad():
        for batch, (guids, images, texts, _) in enumerate(test_dataloader):
            images = images.cuda()
            texts = {name: tensor.squeeze(1).cuda() for name, tensor in texts.items()}
            outputs = model(texts, images)
            _, predicted = torch.max(outputs, dim=1)
            Guids.extend(list(guids))
            predictions.extend(predicted.cpu().numpy().tolist())  
    return Guids,predictions

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    print("starting loading model")
    model = MultimodalModel(pretrained_TextModel=args.pretrained_text,
                            pretrained_ImageModel=args.pretrained_image,
                            fusion_method=args.fusion_method).cuda()
    model.load_state_dict(torch.load(args.weights))

    print("starting loading data")
    test_dataloader=test_data_loader(args.test_data)

    guids, predictions=predict(model,test_dataloader)
    test_output(args.save_path, guids, predictions)


if __name__ == '__main__':
    main()
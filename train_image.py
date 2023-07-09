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
from dataloader import ImageDataloader
from sklearn.metrics import accuracy_score
from model import ImageModel
from tqdm import tqdm
from utils import *
import wandb
import logging
import argparse

# set args
parser = argparse.ArgumentParser("Image-Sentiment-Analysis")
parser.add_argument('--train_data', type=str, default="data/train.txt", help='path of input training data')
parser.add_argument('--data_folder', type=str, default="data/data", help='path of input training data folder')
parser.add_argument('--image_model', type=str, default="resnet", help='model used for images')
parser.add_argument('--pretrained_image', type=str, default="pretrained/resnet/resnet34.pth",
                    help='path of pretrained image model')
parser.add_argument('--validation_size', type=float, default=0.2, help='the size of the validation set')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default="0", help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--save', type=str, default="EXP", help='location of the data corpus')
parser.add_argument('--wandb_id', type=str, default="", help='weight & bias id')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if len(args.wandb_id) != 0:
    wandb.init(
        project="Image Sentiment Analysis using " + args.image_model,
        config=args,
        entity=args.wandb_id
    )

def evaluate(model, valid_dataloader):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for batch, (images, labels) in enumerate(valid_dataloader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)  
            val_pred.extend(predicted.cpu().numpy().tolist())
            val_true.extend(labels.cpu().numpy().tolist())
    accuracy = accuracy_score(val_true, val_pred)
    return accuracy

# 训练过程
def train_and_eval(model, train_dataloader, valid_dataloader, optimizer,scheduler):
    best_acc=0
    criterion = nn.CrossEntropyLoss()
    model.train()
    for i in range(args.epochs):
        start = time.time()
        model.train()
        print("Running training epoch {} ".format(i + 1))
        total_loss = 0.0
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (batch + 1) % (len(train_dataloader) // 10) == 0:
                logging.info("epoch {:04d} step {:04d}/{:04d} loss {:.4f}".format(
                    i + 1, batch + 1, len(train_dataloader), loss))
        logging.info("epoch {:04d}  average_loss {:.4f} | time {:.4f}".format(
            i + 1, total_loss / (batch + 1), time.time() - start))
        
        # evaluation and saving the best model
        model.eval()
        acc = evaluate(model, valid_dataloader)
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(args.save, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            
        logging.info("current acc is {:.4f}, best acc is {:.4f}".format(acc, best_acc))
        logging.info("time costed = {}s \n".format(round(time.time() - start, 5)))



# 主函数
def main():
    # create logging directory
    args.save = os.path.join(args.save, 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
    create_exp_dir(args.save)

    # set logging configuration
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # check the cuda devices
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                     
    # load data
    print("start loading data")
    data_loader=ImageDataloader(args.train_data,
                                args.data_folder,
                                transform, 
                                batch_size=args.batch_size, 
                                validation_size=args.validation_size)
    
    train_dataloader,valid_dataloader = data_loader()

    # load model
    print("start loading model")
    model = ImageModel(pretrained_ImageModel=args.pretrained_image).cuda()
    optimizer = AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader),
                                                num_training_steps=args.epochs * len(train_dataloader))

    # count model parameters
    num_of_parameters = count_parameters(model)
    logging.info("model parameters:{}".format(num_of_parameters))

    # start training
    print("start training")
    train_and_eval(model, train_dataloader, valid_dataloader, optimizer, scheduler)

if __name__ == '__main__':
    main()

import torch
from torch import nn
import numpy as np
from torchvision.models import resnet50,resnet34,resnet18
from transformers import RobertaModel, RobertaConfig
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class TextModel(nn.Module):
    def __init__(self,pretrained_TextModel,num_classes=3):
        super(TextModel, self).__init__()
        self.pretrained_TextModel = pretrained_TextModel
        self.config = RobertaConfig.from_pretrained(self.pretrained_TextModel)
        self.text_model = RobertaModel.from_pretrained(self.pretrained_TextModel)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
    
    def forward(self,text):
        text_outputs = self.text_model(**text)
        pooled_output = text_outputs.pooler_output
        outputs = self.classifier(pooled_output)
        return outputs

class ImageModel(nn.Module):
    def __init__(self,pretrained_ImageModel,num_classes=3):
        super(ImageModel, self).__init__()
        self.pretrained_ImageModel = pretrained_ImageModel
        self.image_model = resnet34()
        self.image_model.load_state_dict(torch.load(self.pretrained_ImageModel))
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, num_classes)
    
    def forward(self,image):
        outputs = self.image_model(image)
        return outputs

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        Q = self.query(inputs)
        K = self.key(inputs)
        V = self.value(inputs)
        attention_scores = torch.matmul(Q, K.transpose(1, 0))
        attention_weights = self.softmax(attention_scores)
        weighted_output = torch.matmul(attention_weights, V)
        return weighted_output

class MultimodalModel(nn.Module):
    def __init__(self, pretrained_TextModel,pretrained_ImageModel,fusion_method,num_classes=3):
        super(MultimodalModel, self).__init__()
        self.pretrained_TextModel = pretrained_TextModel
        self.pretrained_ImageModel = pretrained_ImageModel
        self.config = RobertaConfig.from_pretrained(self.pretrained_TextModel)
        self.text_model = RobertaModel.from_pretrained(self.pretrained_TextModel)
        self.image_model_name=pretrained_ImageModel[-12:-4]
        if self.image_model_name=='resnet50':
            self.image_model = resnet50()
        elif self.image_model_name=='resnet34':
            self.image_model = resnet34()
        else:
            self.image_model = resnet18()
        self.image_model.load_state_dict(torch.load(self.pretrained_ImageModel))
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, self.config.hidden_size)
        self.fusion_method=fusion_method
        if self.fusion_method=='attention':
            self.self_attention = SelfAttention(self.config.hidden_size*2)
        if self.fusion_method=='add':
            self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        elif self.fusion_method=='concat':
            self.classifier = nn.Linear(self.config.hidden_size*2, num_classes)

    def forward(self, text, image):
        text_outputs = self.text_model(**text)
        image_outputs = self.image_model(image)
        pooled_output = text_outputs.pooler_output
        if self.fusion_method=='add':
            outputs = pooled_output + image_outputs
        elif self.fusion_method=='concat':
            outputs = torch.cat([pooled_output, image_outputs], dim=1)
        elif self.fusion_method=='attention':
            outputs = torch.cat([pooled_output, image_outputs], dim=1)
            outputs = self.self_attention(outputs)
        outputs = self.classifier(outputs)
        return outputs

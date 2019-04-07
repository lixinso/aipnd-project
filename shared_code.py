
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from PIL import Image
from collections import OrderedDict

import json
import json

import time
import os

import numpy as np
from collections import OrderedDict

import time


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {"train" : transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
                   "test" : transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]),
                   "valid" : transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
                  }

# TODO: Load the datasets with ImageFolder
image_datasets = {"train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
                 "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["test"]),
                 "test": datasets.ImageFolder(test_dir, transform=data_transforms["valid"])}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
               "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64, shuffle=True),
               "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=64, shuffle=True)}



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
print(len(cat_to_name))
print(json.dumps(cat_to_name,indent=4)[:300])

#Params
hidden_units = 500
learning_rate = 0.001
class_to_idx = image_datasets['train'].class_to_idx

checkpoint_path = 'densenet121_checkpoint.pth'


# TODO: Build and train your network

def get_densenet_model(hidden_units):
    model = models.densenet121(pretrained=True)

   # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier, ensure input and output sizes match
    classifier_input_size = model.classifier.in_features
    classifier_output_size = 102
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, classifier_output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model

def create_model(learning_rate, hidden_units, class_to_idx):
    ''' Create a deep learning model from existing PyTorch model.
    '''
    # Load pre-trained model
    model = get_densenet_model(hidden_units)

    # Set training parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)
    criterion = nn.NLLLoss()

    # Save class to index mapping
    model.class_to_idx = class_to_idx

    return model, optimizer, criterion



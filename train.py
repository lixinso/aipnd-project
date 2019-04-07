# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

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




#import train



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

model, optimizer, criterion = create_model(learning_rate, hidden_units, class_to_idx)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


epochs = 1
steps = 0
running_loss = 0
print_every = 5
trainloader = dataloaders["train"]
testloader = dataloaders["test"]
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    #Accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    
                    
            print(f"Epoch {epoch+1} / {epochs}...)"
                  f"Train loss: {running_loss / print_every:.3f}"
                  f"Test loss: {test_loss / len(testloader):.3f}"
                  f"Test accuracy: {accuracy/len(testloader):.3f}"
                 )
            
            
            #Temp break to avoid running to long. Will remove it later after test
            if accuracy/len(testloader) > 0.71:
                break
                
       
# TODO: Do validation on the test set


accuracy = 0
model.eval()

validloader = dataloaders["valid"]
with torch.no_grad():
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)


        #Accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Valid accuracy: {accuracy/len(testloader):.3f}")




model.class_to_idx = image_datasets['train'].class_to_idx


# Save the checkpoint 
checkpoint_path = 'densenet121_checkpoint.pth'

state = {
    'arch': 'densenet121',
    'learning_rate': learning_rate,
    'hidden_units': hidden_units,
    'epochs': epochs,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'class_to_idx' : model.class_to_idx
}

torch.save(state, checkpoint_path)

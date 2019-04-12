import argparse


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
import time
import os
import sys

import numpy as np
from collections import OrderedDict

import shared_code




model_type="densenet" #densenet,vgg16 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: Write a function that loads a checkpoint and rebuilds the model



# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(model_type):
    
    # Get class to index mapping
    class_to_idx = shared_code.image_datasets['train'].class_to_idx

    if model_type == "densenet":
        
        checkpoint_path = 'densenet121_checkpoint.pth'
        state = torch.load(checkpoint_path)

        learning_rate = state['learning_rate']
        class_to_idx = state['class_to_idx']
        hidden_units = state['hidden_units']

        # Load pretrained model
        model_load, optimizer, criterion = shared_code.create_model(model_type, learning_rate, hidden_units, class_to_idx)

        # Load checkpoint state into model
        model_load.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

        print("Loaded '{}' (arch={}, hidden_units={}, epochs={})".format(
            checkpoint_path, 
            state['arch'], 
            state['hidden_units'], 
            state['epochs']))

        return [model_load,optimizer]

    elif model_type == "vgg16":
        checkpoint_path = "vgg16_checkpoint.pth"
        state = torch.load(checkpoint_path)

        learning_rate = state['learning_rate']
        class_to_idx = state['class_to_idx']

        print("class_to_idx", class_to_idx)
        
        # Load pretrained model
        model_load, optimizer, criterion = shared_code.create_model(model_type, learning_rate, -1, class_to_idx)

        # Load checkpoint state into model
        model_load.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])

        #print("Loaded '{}' (arch={}, hidden_units={}, epochs={})".format(
        print("Loaded '{}' (arch={}, epochs={})".format(
            checkpoint_path, 
            state['arch'], 
            #state['hidden_units'], 
            state['epochs']))

        return [model_load,optimizer]        
    
[model_load,optimizer] = load_checkpoint(model_type)





def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    '''
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    image = np.array(image)
    image = image/255
    
    mean = np.array([0.485, 0.256, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    
    image = np.transpose(image, (2,0,1))
    
    return image.astype(np.float32)
    '''
    
    expects_means = [0.485, 0.256, 0.406]
    expects_std = [0.229, 0.224, 0.225]
    
    pil_image = Image.open(image).convert("RGB")
    
    in_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(expects_means, expects_std)])
    pil_image = in_transforms(pil_image)
    
    return pil_image
    





def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    #img = Image.open(image_path)
    #img = process_image(img)
    img = process_image(image_path)
    
    #2D image to 1D vector
    img = np.expand_dims(img, 0)
    
    img = torch.from_numpy(img)
    
    model.eval()
    
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)


# TODO: Display an image along with the top 5 classes

def view_classify(image, ps, classes):
    num_classes = len(ps)
    ps = np.array(ps)
    ##image = image.transpose((1,2,0))
    
    ##mean = np.array([0.485, 0.456, 0.406])
    ##std = np.array([0.229,0.224, 0.225])
    ##image = std * image + mean
    
    ##image = np.clip(image, 0, 1)
    
    fig, (ax1, ax2) = plt.subplots(figsize=(10,6), ncols=2)
    ax1.imshow(image)
    ax1.axis('off')
    ax2.barh(np.arange(num_classes), ps)
    
    ax2.set_yticks(np.arange(num_classes))
    ax2.set_yticklabels(np.arange(num_classes).astype(int), size='large')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    ax2.set_yticklabels(classes)
    fig.subplots_adjust(wspace=0.6)
    
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--img_path', default='./flowers/train/102/image_08001.jpg', type=str)
    parser.add_argument('--device', default='gpu', type=str) # gpu|cpu
    parser.add_argument('--model_type', default='densenet', type=str) # densenet | vgg16 
    parser.add_argument('--checkpoint_path', default='densenet121_checkpoint.pth', type=str)  #densenet121_checkpoint.pth | vgg16_checkpoint.pth
    parser.add_argument('--top_k', default=5, type=int)
    parser.add_argument('--category_names_path', default='cat_to_name.json', type=str)

    args = parser.parse_args()
    
    img_path             = args.img_path
    device               = args.device
    model_type           = args.model_type
    checkpoint_path      = args.checkpoint_path
    top_k                = args.top_k
    category_names_path  = args.category_names_path
    
    
    img_path = "./flowers/train/102/image_08001.jpg"
    


    #if len(sys.argv) >= 2:
    #    img_path = sys.argv[1]
    
    img = Image.open(img_path)
    img

    probs, classes = predict(img_path, model_load.to(device))
    print(probs)
    print(classes)

    #class_names = shared_code.image_datasets['train'].classes
    #flower_names = [shared_code.cat_to_name[class_names[e]] for e in classes]
    #print(flower_names)
    
    # Reverse the dict
    idx_to_class = {val: key for key, val in model_load.class_to_idx.items()}
    # Get the correct indices
    top_classes = [idx_to_class[each] for each in classes]
    flower_names = [shared_code.cat_to_name[e] for e in top_classes]
    print(flower_names)
    

    #view_classify(img, probs, flower_names)

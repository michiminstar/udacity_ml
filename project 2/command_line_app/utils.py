import os
import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

def create_loaders(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(degrees=25),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                                for x in ['train', 'valid', 'test']}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                            batch_size=32,
                            shuffle=True) for x in ['train', 'valid', 'test']}
   
    return dataloaders, image_datasets



def process_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    
    if width < height:
        resize_size = (256, 256**600)
    else:
        resize_size = (256**600, 256)
    
    image.thumbnail(resize_size)
    CROP_SIZE = 224
    
    # Crop Images
    left = width / 4 - CROP_SIZE / 2
    top = height / 4 - CROP_SIZE / 2
    right = width / 4 + CROP_SIZE / 2
    bottom = height / 4 + CROP_SIZE / 2
    image = image.crop((left, top, right, bottom))
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = np.array(image) / 255
    
    image = (image_array - mean) / std
    image = image.transpose((2,0,1))
    
    return torch.from_numpy(image)
    
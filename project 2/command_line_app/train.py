'''
    # TODO
    1. function for data preprocessing (utils.py)
    2. create a function that creates  PyTorch dataloaders (utils.py)
    3. create a function for initializing a model using a pre-trained model (classifier.py)
    4. create a function for training, evaluation, and prediction  (functions.py)
    5. create a function for saving and loading checkpoint (functions.py)
'''

import argparse
import torch
from classifier import initialize_model
from utils import create_loaders
from functions import train_model, test_model, save_checkpoint

def main():
    parser = argparse.ArgumentParser(description='This program predicts a flower name from an image')
    parser.add_argument('data_dir', type=str, help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='./', help='Saved checkpoint directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='Network architecture')
    parser.add_argument('--hidden_units', type=int, default='256', help='Hidden units')
    parser.add_argument('--dropout', type=float, default='0.2', help='Dropout for the hidden layers')
    parser.add_argument('--num_classes', type=int, default='256', help='Number of classes for classification')
    parser.add_argument('--learning_rate', type=float, default='0.005', help='Learning rate')
    parser.add_argument('--epochs', type=int, default='20', help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU', default=False)
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'
    
    model, criterion, optimizer, scheduler = initialize_model(args.arch, args.hidden_units, args.dropout, args.num_classes, device, args.learning_rate)
    
    dataloaders, image_datasets = create_loaders(args.data_dir)
    
    train_model(model, dataloaders, criterion, optimizer, scheduler, image_datasets, args.epochs, device)
    test_model(model, dataloaders, image_datasets, device)
    save_checkpoint(model, optimizer, scheduler, args.epochs, args.learning_rate, f'{args.arch}_checkpoint.pth')
            
if __name__ == '__main__':
    main()
    
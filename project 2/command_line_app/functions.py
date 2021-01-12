import os
import copy
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

def train_model(model, dataloaders, criterion, optimizer, scheduler, image_datasets, num_epochs, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs = num_epochs

    print('Training process starting .....\n')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
 
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # attach indices of classes to the model
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    return model


def test_model(model, dataloaders, image_datasets, device):
    print("\nRunning model on test set...")
        
    avg_acc = 0
    acc_test = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:      
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            acc_test += torch.sum(preds == labels.data).item()

    avg_acc = acc_test / len(image_datasets['test'])

    print('Accuracy on test images: %d%%' % (100 * avg_acc))

def predict(model, image_path, cat_to_name, topk=5):
    model.to('cuda')
    model.eval()
    
    image = image_path.cuda().float()
    image = image.unsqueeze_(0)
    
    idx_to_class = {val:key for key, val in model.class_to_idx.items()}

    with torch.no_grad():
        output = model.forward(image)
        top_probs, top_classes = torch.topk(output, topk)        
        idx = np.array(top_classes)[0]
        
        classes = [idx_to_class[i] for i in idx]

        names = [cat_to_name[idx_to_class[i]] for i in idx]

    return top_probs, classes, names

def save_checkpoint(model, optimizer, scheduler, epochs, learning_rate, filename='checkpoint.pth'):
    print(f"Saving model..")

    checkpoint = {
        'model': model,
        'state_dict': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
        'scheduler': scheduler.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs,
        'learning_rate': learning_rate,
    }

    torch.save(checkpoint, filename)


def load_checkpoint(device, filename='checkpoint.pth'):
    print("=> loading checkpoint '{}'".format(filename))

    checkpoint = torch.load(filename + '.pth')

    model = checkpoint['model']
    model = model.to(device)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    if model == 'vgg' or model == 'alexnet':
        optimizer = optim.SGD(model.classifier.parameters(),
                              lr=(checkpoint['learning_rate'] if 'learning_rate' in checkpoint else 0.005), momentum=0.9)
    else:
        optimizer = optim.SGD(model.fc.parameters(),
                                   lr=(checkpoint['learning_rate'] if 'learning_rate' in checkpoint else 0.005), momentum=0.9)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epochs']

    model.train()
    
    return model, optimizer, scheduler, epoch

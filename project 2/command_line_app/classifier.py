import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models

'''
    reference:
    *check initialize_model function
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#inception-v3
'''

def initialize_model(arch, hidden_units, dropout, num_classes, device, learning_rate=0.005):
    # get model attributes of an architecture
    model = getattr(models, arch)(pretrained=True)

    # freeze the param for training
    for param in model.parameters():
        param.requires_grad = False

    if arch == 'vgg16' or arch == 'alexnet':
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1)
        )
    elif arch == 'resnet18' or arch == 'resnet34':
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1)
        )
    else:
        print('Architecture not available. Applying the default: vgg16... ')
        print('If you want, you can try alexnet, resnet18, or  resnet34 instead.')

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1)
        )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = learning_rate

    if arch == 'vgg16' or arch == 'alexnet':      
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model.classifier[6].parameters(), lr=learning_rate, momentum=0.9)
    elif arch == 'resnet18' or arch == 'resnet34':
        optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)
    else:
          exit()

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    return model, criterion, optimizer, scheduler


import os
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

from collections import OrderedDict

### python train.py ./flowers


def dataLoaders(data_dir):
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'
    
    data_transforms = {
        'training' : transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225]
                )
            ]
        ),

        'validation' : transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225]
                )
            ]
        ),

        'testing' : transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225]
                )
            ]
        )
    }
    
    image_datasets = {
        'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }

    data_loaders = {
        'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=False),
        'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True)
    }
    
    class_to_idx = image_datasets['training'].class_to_idx
    return data_loaders, class_to_idx

def generate_model(arch, hidden_units, learning_rate):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'densenet201':
        model = models.densenet201(pretrained=True)
        input_size = model.classifier.in_features
    else:
        raise Exception("Unknown model")

    for param in model.parameters():
        param.requires_grad = False

    output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    if 'vgg' in arch:
        model.classifier = classifier
    elif 'densenet' in arch:
        model.classifier = classifier
    elif 'resnet' in arch:
        model.fc = classifier
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)
    optimizer.zero_grad()
    criterion = nn.NLLLoss()
    
    return model, optimizer, criterion

def train(device, model, criterion, optimizer, trainingLoader, validationLoader, epochs, print_every):                
    steps = 0
    len_validation_loader = len(validationLoader)
    
    for e in range(epochs):
        running_loss = 0
                
        print('-' * 20)
        print('Epoch: {} / {}'.format(e+1, epochs))
                
        for inputs, labels in iter(trainingLoader):
            steps += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # For debug
            #print("{}/{}".format(steps, print_every))
            
            if steps % print_every == 0:
                validation_loss, accuracy = validate(device, model, criterion, validationLoader)
                print('  Training Loss: {:.3}'.format(running_loss / print_every), 
                      ', Validation Loss: {:.3}'.format(validation_loss / len_validation_loader),
                      ', Validation Accuracy: {:.2f}%'.format(accuracy / len_validation_loader * 100))                
                running_loss = 0

    
def validate(device, model, criterion, dataLoader):
    data_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in dataLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            data_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    return data_loss, accuracy


def main():
    architectures = {
        'vgg13',
        'vgg16',
        'vgg19',
        'densenet121',
        'densenet161',
        'densenet201'
    }    
    parser = argparse.ArgumentParser()    
    parser.add_argument('--arch', dest='arch', default='vgg16', action='store', choices=architectures)  
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=3)

    args = parser.parse_args()
    
    use_gpu = torch.cuda.is_available() and args.gpu

    dataloaders, class_to_idx = dataLoaders(args.data_dir)
    model, optimizer, criterion = generate_model(args.arch, args.hidden_units, args.learning_rate)
    model.class_to_idx = class_to_idx
    
    if use_gpu:
        model.cuda()
        criterion.cuda()

    device = torch.device("cuda" if use_gpu else "cpu")
    model.to(device)
    
    print('current device: ', device)
    print('-' * 20)
    print('Training Start ...')
    
    train(
        device,
        model, 
        criterion, 
        optimizer, 
        dataloaders['training'], 
        dataloaders['validation'],
        args.epochs, 
        30
    )
    
    print('-' * 20)
    print('Traning End!!')
    
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_path = args.save_dir + '/' + args.arch + 'checkpoint.pth'
    else:
        save_path = args.arch + 'checkpoint.pth'

    model.class_to_idx = class_to_idx
    object = {
        'arch': args.arch,
        'learning_rate': args.learning_rate,
        'hidden_units': args.hidden_units,
        'class_to_idx': model.class_to_idx,
        'num_of_epochs': args.epochs,
        'optimizer': optimizer.state_dict(),
        'classifier': model.classifier,
        'state_dict': model.state_dict()
    }
    filepath = 'checkpoint.pth'
    torch.save(object, filepath)

    print('current device: ', device)
    print('-' * 20)
    print('Testing Start ...')
    len_testing_loader = len(dataloaders['testing'])
    testing_loss, accuracy = validate(device, model, criterion, dataloaders['testing'])
    print('  Testing Loss: {:.3}'.format(testing_loss / len_testing_loader),
          ', Testing Accuracy: {:.2f}%'.format(accuracy / len_testing_loader * 100))
    print('Testing End ...')
    
if __name__ == "__main__":
    main()
"""train.py

This provides a CLI for training a neural network for image recognition.

Usage:
    >> python train.py -s . -d ./flowers -g -e 20
    * -s : save checkpoint 
    * -d ./flowers : location of the data directory
    * -g : use GPU
    * -e 20 : use 20 epochs

"""
#%% Imports
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime
import numpy as np
from pathlib import Path
from torch import nn, exp
import matplotlib.pyplot as plt
import torch
# from torch import optim, utils, nn, exp, FloatTensor, max, save, load
from torchvision import datasets, models, transforms
from time import time as timer

#%% Parse arguments
parser = ArgumentParser(description=__doc__,
    formatter_class=RawDescriptionHelpFormatter
)
# Architecture
arch_map = {
    'alexnet': models.alexnet
    ,'AlexNet': models.AlexNet
    ,'resnet': models.resnet
    ,'ResNet': models.ResNet
    ,'resnet18': models.resnet18
    ,'resnet34': models.resnet34
    ,'resnet50': models.resnet50
    ,'resnet101': models.resnet101
    ,'resnet152': models.resnet152
    ,'vgg': models.vgg
    ,'VGG': models.VGG
    ,'vgg11': models.vgg11
    ,'vgg11_bn': models.vgg11_bn
    ,'vgg13': models.vgg13
    ,'vgg13_bn': models.vgg13_bn
    ,'vgg16': models.vgg16
    ,'vgg16_bn': models.vgg16_bn
    ,'vgg19_bn': models.vgg19_bn
    ,'vgg19': models.vgg19
    ,'squeezenet': models.squeezenet
    ,'SqueezeNet': models.SqueezeNet
    ,'squeezenet1_0': models.squeezenet1_0
    ,'squeezenet1_1': models.squeezenet1_1
    ,'inception': models.inception
    ,'Inception3': models.Inception3
    ,'inception_v3': models.inception_v3
    ,'densenet': models.densenet
    ,'DenseNet': models.DenseNet
    ,'densenet121': models.densenet121
    ,'densenet169': models.densenet169
    ,'densenet201': models.densenet201
    ,'densenet161': models.densenet161    
}
# Parser arguments
add = parser.add_argument
# Positional

add('-a', '--arch', type=str, default='vgg16_default'
    ,help='Architecture: Pick model from torch.models')
# Checkpoint
add('-c', '--checkpoint', type=str, default=None
    ,help='Optional checkpoint file')
add('-d', '--data_dir', action='store', help='Data directory (must have "train", "test", and "valid" subdirectories)')
# Epochs
add('-e', '--epochs', type=int, default='-1', help='Number of epochs')
# GPU
add('-g', '--gpu', action='store_true', help='Use GPU if available')
# Learning rate
add('-r','--learning_rate', type=float, default=-1.0
    ,help='Learning rate')
# Save directory
add('-s', '--save_dir', type=str, required=True
    ,help='Save: path to save directory (**required**)')
# Hidden units
add('-u', '--hidden_units', metavar='N'
    ,help='List of hidden layers'
    ,default=[-1])
args = parser.parse_args()

#%% Class definitions
class Model:
    def __init__(self):
        # Architecture
        if args.arch=='vgg16_default':
            self.arch = 'vgg16'
            print('WARN: No architecture (-a, --arch) specified. Using arch = '+str(self.arch))
            
        else:
            err_msg = '\n\nSelect an architecture (-a, --arch) from the following choices: '
            if args.arch in arch_map:
                self.arch = args.arch
            else:
                raise ValueError(err_msg+', '.join(arch_map.keys()))
        # Hidden layers
        if args.hidden_units[0]==-1:
            self.hidden_units = [4096, 256]
            print('WARN: No hidden units specified (-l, --hidden_units). Using '+str(self.hidden_units))
        else:
            self.hidden_units = args.hidden_units
        # GPU
        self.device = 'cuda' if args.gpu else 'cpu'
        # Epochs
        if args.epochs==-1:
            self.epochs = 20
            print('WARN: No epochs (-e, --epochs) specified. Using epochs = '+str(self.epochs))
        else:
            self.epochs = abs(args.epochs)
        # Learning rate
        if args.learning_rate==-1.0:
            self.lr = 0.001
            print('WARN: No learning rate (-l, --learning_rate) specified. Using learning_rate = '+str(self.lr))
        else:
            self.lr = abs(args.learning_rate)
        # Data directory
        p = Path(args.data_dir)
        req_dirs = {'test', 'train','valid'}
        if not (req_dirs <= {x.name for x in p.iterdir() if x.is_dir()}):
            raise ValueError('data_dir must contain subdirectories: train, test, valid')
        else:
            self.path = p
        
        # Defaults
        self.phases = 'train test valid'.split(' ')
        self.set_transforms()
        self.set_data_loaders()
        
    def set_transforms(self, train=None, other=None):
        train_transforms = transforms.Compose([
            transforms.RandomRotation(45)
            ,transforms.RandomHorizontalFlip()
            ,transforms.RandomResizedCrop(224)
            ,transforms.ToTensor()
            ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) if train==None else train

        other_transforms = transforms.Compose([
            transforms.Resize(256)
            ,transforms.RandomResizedCrop(224)
            ,transforms.ToTensor()
            ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) if other==None else other

        self.image_transforms = {
            'train': train_transforms
            ,'test': other_transforms
            ,'valid': other_transforms
        }
    
    def set_data_loaders(self, batch_size=32):
        #%% Create datasets and loaders
        self.image_datasets = {
            mode: datasets.ImageFolder(self.path / mode, transform=self.image_transforms[mode])
            for mode in self.phases
        }

        self.data_loaders = {
            mode: torch.utils.data.DataLoader(self.image_datasets[mode], batch_size, shuffle=True)
            for mode in self.phases
        }
    
    def set_classnames(self, class_to_label=None):
        c2l = self.path / class_to_label
        if c2l.exists():
            with open(c2l, 'r') as f:
                self.class_to_label = json.load(f)
    
    def set_model(self, num_input=25088, num_output=102, drop_rate=0.35):
        self.model = arch_map[self.arch](pretrained=True)
        self.model.class_to_idx = self.image_datasets['train'].class_to_idx
        for param in self.model.parameters():
            param.require_grad = False
        # Define input -> first layer
        classifier = torch.nn.Sequential()
        cadd = classifier.add_module # alias
        cadd("input", torch.nn.Linear(num_input, self.hidden_units[0]))
        cadd("rel0", torch.nn.ReLU())
        cadd("dp0", torch.nn.Dropout(drop_rate))
        hu = self.hidden_units # alias
        if len(self.hidden_units) > 1:
            for i,d in enumerate(hu[:-1]):
                cadd("hl"+str(i+1), torch.nn.Linear(hu[i], hu[i+1]))
                cadd("rel"+str(i+1), torch.nn.ReLU())
                cadd("dp"+str(i+1), torch.nn.Dropout(drop_rate))
        cadd("final", torch.nn.Linear(hu[-1], num_output))
        cadd("output", torch.nn.LogSoftmax(dim=1))

        # Replace the model's classifier with mine
        self.model.classifier = classifier
        
    def train(self, criterion_func=torch.nn.NLLLoss, optim_func=torch.optim.Adam):
        criterion = criterion_func()
        optimizer = optim_func(self.model.classifier.parameters(), self.lr)
        start = timer()
        device = self.device # alias
        self.model.to(device)
        for e in range(self.epochs):
            train_loss, test_loss, accuracy = 0, 0, 0
            # Train
            print("Epoch: {}/{}".format(e+1, self.epochs), end=' steps.. ', flush=True)
            self.model.train()
            for images, labels in self.data_loaders['train']:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print("Training Loss: {:.3f}".format(train_loss/len(self.data_loaders['train'])), end='.. ')
            # Test
            self.model.eval()
            with torch.no_grad():
                for images, labels in self.data_loaders['test']:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model.forward(images)
                    test_loss += criterion(outputs, labels).item()
                    ps = torch.exp(outputs)
                    equality = (labels.data == ps.max(dim=1)[1])
                    accuracy += equality.type(torch.FloatTensor).mean()

            print(
                "Test Loss: {:.3f}.. ".format(
                    test_loss/len(self.data_loaders['test'])
                )
                ,"Accuracy: {:.3f}.. ".format(accuracy/len(self.data_loaders['test']))
                ,"Elapsed: {:.2f} mins".format((timer()-start)/60)
            )
    
    def validate(self):
        total_match, total = 0, 0
        self.model.eval()
        torch.no_grad()
        device = self.device # shorthand
        for images, labels in self.data_loaders['valid']:
            images, labels = images.to(device), labels.to(device)
            outputs = self.model.forward(images)
            _, prediction = torch.max(outputs.data, 1)
            total_match += (labels == prediction).sum().item()
            total += labels.size(0)
        print('Validation Accuracy {}'.format(total_match / total))
    
    def checkpoint(self):
        nowish = datetime.now().strftime('%Y%m%d_%H%M')
        file_name = 'checkpoint_'+nowish+'.pth'
        save_file = self.save_dir
        if save_file[-4:]!='.pth':
            if save_file[-1]=='/': save_file += '/'
            save_file += file_name
        torch.save({
            'arch': self.arch
            ,'state_dict': self.model.state_dict()
            ,'classifier': self.model.classifier
            ,'class_to_idx': self.model.class_to_idx
        }, save_file)
        print('Saving model to {}'.format(save_file))
              
    def load(self, checkpoint_file):
        chkpnt = torch.load(checkpoint_file)
        if chkpnt['arch']!=self.arch:
            print('Error: {} not loaded because it did not match arch = {}'.format(
                checkpoint_file
                ,self.arch
            ))
            return False
        else:
            self.model = arch_map[self.arch](pretrained=True)
            self.model.class_to_idx = self.image_datasets['train'].class_to_idx
            for param in self.model.parameters():
                param.require_grad = False
            self.model.classifier = chkpnt['classifier']
            self.model.load_state_dict(chkpnt['state_dict'])
            self.model.class_to_idx = image_datasets['train'].class_to_idx
            return True

#%% Train a model
model = Model()
if args.checkpoint==None:
    model.set_model()
else:
    model.load(args.checkpoint)
model.train()
model.validate()
model.checkpoint()

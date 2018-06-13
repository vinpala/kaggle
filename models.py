import config
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision.models as M
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import copy
import arch
from config import NUM_CLASSES
from arch.resnext import resnext101_32x4d, resnext101_64x4d

class Model(nn.Module):
    '''
      Class for ConvNet(usually pretrained on Imagenet)
    
    '''
    def __init__(self, num_classes=NUM_CLASSES, net_name=None, use_gpu=True, freeze_layers=True, pretrained=True, dropout=False):

        super().__init__()
        
        if net_name not in ['resnet18' , 'densenet', 'resnext', 'resnet101']:
            raise ValueError("This model needs to be loaded") 
     
        if net_name == 'resnet18':         
            self.net = M.resnet18(pretrained=pretrained)
            self.feature_layers_num = 8
        elif net_name == 'densenet121': 
            self.net = M.densenet121(pretrained=pretrained)
        elif net_name == 'resnext':
            self.net = resnext101_64x4d(pretrained='imagenet')
        elif net_name == 'resnet101': 
            self.net = M.resnet101(pretrained=pretrained)
        elif net_name == 'resnet34mod':
            self.net = M.resnet34(pretrained=pretrained)
            self.feature_layers_num = 8
            
# By default we use ConvNet as fixed feature extractor(freeze_layers = True)
#we need to freeze all the network except the final layer. We need to set requires_grad == False to freeze the parameters so that the #gradients are not computed in backward().
        self.net_name = net_name
        if freeze_layers:
        # Parameters of newly constructed modules have requires_grad=True by default
            for param in self.net.parameters(): param.requires_grad = False
           
        if 'densenet' in net_name:
            self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)
            self.optim_params = self.net.classifier.parameters()
        else:
            if net_name == 'resnet34mod':
# replace avgpool with special kind of average-pooling for which you specify the output activation resolution rather 
#than how big of an area to poll.Adding adaptive max pool on top of this gives more flexibility
                self.net = nn.Sequential(*self.net.children()[:-2], 
                  nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1), 
                  nn.Linear(self.net.fc.in_features, num_classes))
                self.optim_params = self.net.children()[-1].parameters()
            else:
                self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
                self.optim_params = self.net.fc.parameters()
        #print("optimum",self.optim_params)
        #if use_gpu:
            #self.net = self.net.cuda()
            
    def finetune(self, freeze_until, freeze_layers_ct):
        '''
         freeze_until : finetune(value none) or name of the layer (eg: unfreeze all layers after "Conv2d_4a_3*3")
         freeze_layers_ct : no of initial layers to be frozen 
        '''
        #Unfreeze all 
        if freeze_until is None:
            for i, param in self.net.parameters():
                param.requires_grad = True
        # All parameters are being optimized
            self.optim_params = self.net.parameters()
            return
        elif freeze_layers_ct is None: 
            unfreeze = False
            for name, child in self.net.named_children():
                if unfreeze:                
                     for params in child.parameters():          
                            params.requires_grad = True
                if freeze_until in name:
                    unfreeze = True
        else:  
            ct = 0
            for name, child in self.net.named_children():
                ct += 1
                if ct > freeze_layers_ct:
                    for name2, params in child.named_parameters():
                        params.requires_grad = True
            
        self.optim_params = list(filter(lambda p: p.requires_grad, self.net.parameters()))
    
    def get_net(self):
        return self.net
    
    def get_layers(self):
        return list(self.get_net().children())
                    
    def get_optim_params(self, layers=None):
        
        if layers is None:
            # parameters of all layers needs to be optimized
            self.optim_params = list(filter(lambda p: p.requires_grad, self.net.parameters()))
            return  self.optim_params
        params_grad = []
        
        for layer in layers:
            for param in layer.parameters():
                if param.requires_grad:
                    params_grad.append(param)
        return params_grad
        
    def display(self):
        for name, child in self.net.named_children():
            for name_2, params in child.named_parameters():
                print(name+' '+name_2 +'  : '+ 'trainable' if params.requires_grad else (name+' '+name_2 +'  : '+ 'non-trainable'))
    
                
    def forward(self, x):
        return self.net(x)
    
    def set_layer_groups(self, start_layers):
        '''
           
          start_layers : list of layer numbers for segregating the net into layer-groups
          
          output: layer_groups
          
        '''
        layers = self.get_layers()
        layer_groups = []
        last_idx = 0
        for idx in start_layers:           
            layer_groups.append(layers[last_idx:idx])
            last_idx = idx
            
        return layer_groups 
    
class LastLayers(nn.Module):
    def __init__(self, old_model):
        super().__init__()
        self.pre_fc = copy.deepcopy(torch.nn.Sequential(*list(old_model.net.children())[old_model.feature_layers_num :-1]))
        self.fc = copy.deepcopy(torch.nn.Sequential(*list(old_model.net.children())[old_model.feature_layers_num + len(self.pre_fc):]))

    def forward(self, x):
        x = self.pre_fc(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x    

            
        
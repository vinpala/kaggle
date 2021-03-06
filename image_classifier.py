from __future__ import print_function, division
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
#import torchvision
#from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import tqdm
from tqdm import tqdm
import torch.nn.functional as F
from transforms import preprocess, preprocess_hflip, preprocess_with_augmentation
import lr_scheduler_new, models, dataset, error_analysis
from lr_scheduler_new import *
from models import Model
from models import LastLayers
from dataset import *
from error_analysis import *

class Classifier(object):
    
    def __init__(self, title, ds_class_name, batch_size, net_name=None, precompute=False, use_gpu=True, **kwargs):
        
        '''
         Experiment classifier set up 
         
         Parameters -
         title        : short description of experiment
         use_gpu : accelerate run-time using gpus
         out_dir      : directory where model-weights etc will be saved
         
        '''
        if title is None:
            raise ValueError("No Title : Give a meaningful Title")
            
        if len(title) < 10:
            raise ValueError("Short Title : Give a meaningful Title")
            
        if ds_class_name is None :
            raise ValueError("No DatasetClass Specified")
        
        fully_qualified_path = 'dataset'
        p = __import__(fully_qualified_path)

        self.dataset_cls = getattr(p, ds_class_name)

        self.title   = title  
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.out_dir = title
        self.precompute = precompute

        if net_name is None: net_name='resnet18'  
        self.set_model(net_name=net_name)
        
        self.get_data()
        
        if precompute:
            self._precompute_activ(net_name)
        
        #defaults
        if self.use_gpu: # torch.cuda.is_available()
            torch.cuda.synchronize()
            self.model.net = self.model.net.cuda()
            
        self.set_optimizer(opt_name = 'SGD', layer_groups=[], lrs=[0.001], mom=0.9, wd=0, nesterov=False)
        #self.set_lrscheduler(scheduler_name = 'cos-anneal-with-restarts', T_max=1, T_mult=1, take_snapshot=False, gamma=0, step_size=0)
        self.set_criterion(loss_name = 'cross-entropy loss', weight=None)
        self.eval_metric = 'average_error'
        self.time_elapsed = 0
        self.tta = None
        self.metric_val =None
        self.__dict__.update(kwargs)
    
    def __repr__(self): return 'Title = '+ self.title  + ',' + ' GPU-used = ' + str(self.use_gpu) + ',' +\
        ' Batch-size = '+ str(self.batch_size) + ',' +\
        ' Dataset ='+ str(self.dataset_cls) + ',' + \
        ' Dataset-sizes ='+ str(self.dataset_sizes) + ',' +' Out-directory ='+str(self.out_dir) + ',' + ' Model =' +\
        str(self.model.net_name) + ',' +\
        ' Optimizer ='+ str(self.optimizer_name) + ',' +' Scheduler =' + str(self.scheduler_name) + ',' + \
        ' Criterion ='+str(self.criterion_name) + ',' +' Training-time ='+ str(self.time_elapsed) + ',' + ' TTA = '+ str(self.tta) + ','+\
        ' Evaluation metric ='+str(self.eval_metric) + ',' +' Evaluation value ='+str(self.metric_val)
    
    def fit(self, which='all', num_epochs=2, opt_name='SGD', lrs=[], layer_groups=[], wd=0, mom=0.9, nesterov=False, 
            scheduler_name='cos-anneal-with-restarts', T_max=1, T_mult=1, take_snapshot=False, gamma=0, step_size=0, 
            loss_name = 'cross-entropy loss', weight=None):

        '''
        which : 'all'/'last'
               'all' : train all layers
               'last': train just the last layers
        '''

        if which == 'all':
            self.set_optimizer(opt_name, layer_groups, lrs, mom, wd, nesterov)
            self.set_lrscheduler(scheduler_name ,T_max=T_max, T_mult=T_mult, 
                                 take_snapshot=take_snapshot, gamma=gamma, step_size=step_size)
            self.set_criterion(loss_name=loss_name, weight=weight)
            self._train(num_epochs=num_epochs, resumed=False)
        else: 
            if which == 'last': 
                classifier_lyrs = LastLayers(self.model)
                if self.use_gpu:  classifier_lyrs =  classifier_lyrs.cuda()
                self.optimizer = optim.SGD([{'params':classifier_lyrs.fc.parameters(), 'lr':lrs[0]}], 
                                           weight_decay=wd, momentum=mom, nesterov=nesterov)
                self.scheduler = None
                self._train_fc_layers(classifier_lyrs, num_epochs) #train last layers only
                 
    #adapated from http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    def _train(self, num_epochs,resumed=False):
        '''
         forward and backward propagation  
        '''
        #pdb.set_trace()
        print("Starting training")
        
        since = time.time()
             
        if resumed:
            print("Resuming from earlier")
            last_epoch, best_model_wts, min_loss = load_checkpoint(self)
        else:
            last_epoch = 0
            best_model_wts = copy.deepcopy(self.model.state_dict())
            min_loss = 0.0

        for epoch in range(last_epoch, num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if self.scheduler is not None: self.scheduler.step()
                    self.model.train(True)  # Set model to training mode
                else:
                    self.model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

            # Iterate over data.
                #pbar = tqdm(self.dataloaders[phase], total=self.dataset_sizes[phase])
                #for data in pbar:
                # get the inputs
                #    inputs, labels = data
                for inputs, labels in self.dataloaders[phase]:

                # wrap them in Variable
                    if self.use_gpu:
                        torch.cuda.synchronize()
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                    self.optimizer.zero_grad()
    
                # forward
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = self.criterion(outputs, labels)

                # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                # statistics
                    running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                if phase == 'val':
                    if (min_loss == 0.0) or (epoch_loss < min_loss):
                        min_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_checkpoint(epoch, best_model_wts, min_loss)
            print()

        time_elapsed = time.time() - since
        self.time_elapsed = time_elapsed
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        print('Min val loss: {:4f}'.format(min_loss))

    # load best model weights
        self.model.load_state_dict(best_model_wts)
        if self.scheduler: self.plot_lrs()
        return 
    
    def predict(self, data_transforms, phase, analyze_results=False):
        '''
        Predict with Test Time Augmentation (TTA)
        Additional to the original test/validation images, apply image augmentation to them
        (just like for training images) and calculate the mean of predictions. The intent
        is to increase the accuracy of predictions by examining the images using multiple
        perspectives.

        '''
        state = torch.load(self.out_dir+'/checkpoint.pth.tar')
        model = self.model.load_state_dict(state['best_model_wts'])
        #dropout and batchnorm are disabled in the evaluation mode if you have such steps in your model.
        #This sets self.training to False for every module in the model.
        model.eval()
        dataloaders =[]
        self.tta = data_transforms
        for transform in data_transforms[phase]:
            test_dataset = self.dataset_cls(phase, transform=transform)
            dataloader = DataLoader(dataset=test_dataset, num_workers=1, batch_size=self.batch_size, shuffle=False)
            dataloaders.append(data_loader)
        prediction = []
        
        for dataloader in dataloaders:
            all_labels = []
            all_outputs = []
            pbar = tqdm(dataloader, total=len(dataloader))
            for data in pbar:
                inputs, labels = data
                if self.use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                else: inputs, labels = Variable(inputs), Variable(labels)
                outputs = model(inputs)
                all_outputs.append(outputs)
                all_labels.append(labels)
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            if self.use_gpu:
                all_labels = all_labels.cuda()
                all_outputs = all_outputs.cuda()
            if len(prediction) == 0:
                prediction = all_outputs
            else: prediction = torch.stack((prediction, px), dim=-1)
        
        prediction_prob = F.softmax(prediction, dim=1).data.numpy()
        prediction_prob = prediction_prob.mean(axis=2)
        prediction = np.argmax(prediction_prob, axis=1)
        correct = (prediction == all_labels).sum()
        total = all_labels[0]
        average_error = (total - correct)/total 
        print('Total  :',total)
        print('Correct ones :',total)
        print('Average error for this classifier is : {}', average_error)
        self.eval_metric = 'average_error'
        self.metric_val = average_error
        self.save_experiment()
        if analyze_results:
            analyze_results(test_dataset,prediction_prob)
    
    def kaggle_predict(self, data_transforms, submit_file, phase):
        state = torch.load(self.out_dir+'/checkpoint.pth.tar')
        model = self.model.load_state_dict(state['best_model_wts'])
        #dropout and batchnorm are disabled in the evaluation mode if you have such steps in your model.
        #This sets self.training to False for every module in the model.
        model.eval()
        dataloaders =[]
        
        for transform in data_transforms[phase]:
            test_dataset = self.dataset_cls(phase, transform=transform)
            dataloader = DataLoader(dataset=test_dataset, num_workers=1, batch_size=self.batch_size, shuffle=False)
            dataloaders.append(data_loader)
        prediction = []
        
        for dataloader in dataloaders:
            all_outputs = []
            pbar = tqdm(dataloader, total=len(dataloader))
            for inputs in pbar:
                if self.use_gpu:
                        inputs = Variable(inputs.cuda())
                else: inputs = Variable(inputs)
                outputs = model(inputs)
                all_outputs.append(outputs)
            all_outputs = torch.cat(all_outputs)
            if self.use_gpu:
                all_outputs = all_outputs.cuda()
            if len(prediction) == 0:
                prediction = all_outputs
            else: prediction = torch.stack((prediction, px), dim=-1)
        prediction = F.softmax(prediction, dim=1).data.numpy()
        prediction = np.argmax(prediction.mean(axis=2), axis=1)
        sx = pd.read_csv(submit_file)
        sx.loc[sx.id.isin(test_dataset.data.image_id), 'predicted'] = prediction
        sx.to_csv(self.out_dir+'/kaggle_prediction.csv', index=False)
        
    def predict_from_snapshots(self):
        '''
          TODO
        '''
        pass
    
    def set_criterion(self, loss_name, weight=None):
        
            # cross-entropy    
            #It is useful when training a classification problem with n classes.
            #If provided, the optional argument weights should be a 1D Tensor assigning weight to each of the classes. 
            #This is particularly useful when you have an unbalanced training set.
        if loss_name == 'cross-entropy loss': self.criterion = torch.nn.CrossEntropyLoss(weight=weight)

            #MSE loss
            #Creates a criterion that measures the mean squared error between n elements in the input x and target y:
        elif loss_name == 'MSE loss': self.criterion = torch.nn.MSELoss()
            
           # L1 Loss
           #Creates a criterion that measures the mean absolute value of the element-wise difference between input x and target y: 
        else:   
            if loss_name =='L1 loss': self.criterion = torch.nn.L1Loss()
            else:raise ValueError("only 'cross-entropy loss', 'MSE loss' and 'L1 loss'")
        self.criterion_name = loss_name
        print("criterion set as :", loss_name)
        return

    def set_model(self, net_name, use_gpu=True, freeze_layers=True, pretrained=True, dropout=False):
        '''
          Can be a pretrained net or a new one
        '''
        self.model = Model(net_name=net_name, use_gpu=True, freeze_layers=True, pretrained=True, dropout=False)
        #print("model :", self.model)
        return  
    
    def set_lrscheduler(self, scheduler_name ,T_max, T_mult, take_snapshot, gamma, step_size, eta_min=1e-3, last_epoch=-1):
        
        optimizer = self.optimizer 
        if scheduler_name == 'cos-anneal-with-restarts':
            out_dir = self.out_dir
            model = self.model
            self.scheduler = CosineAnnealingLR_with_Restart(optimizer, T_max, T_mult, model, out_dir, take_snapshot, eta_min=eta_min)
        elif scheduler_name == 'cyclic': self.scheduler =  CyclicLR(optimizer)
        elif scheduler_name == 'step': self.scheduler =  optim.StepLR(optimizer, step_size)
        elif scheduler_name == 'exponential': self.scheduler =  optim.ExponentialLR(optimizer, gamma)
        else:
            raise ValueError("LR scheduler shud be one of ['cos-anneal-with-restarts', 'cyclic', 'step', 'exponential']")
        self.scheduler_name = scheduler_name
        print("LR scheduler set up :", self.scheduler)
        return
    
    def set_optimizer(self, opt_name, layer_groups, lrs, mom, wd, nesterov):
        '''
        layer_groups : returns by model.set_layer_groups(layer_start_indices)
        lrs: list of leraning rates to be applied to different layer-groups "differential learning rates"
        fast.ai recommendation - Set earlier layers to 3x-10x lower learning rate than next higher layer

        '''
        if len(layer_groups) == 0: opt_params = [{'params':self.model.get_optim_params(), 'lr':lrs[0]}]
        else:
            group_lr = zip(layer_groups, lrs)
            opt_params = [{'params': self.model.get_optim_params(p[0]), 'lr': p[1]} for p in group_lr]
        
        self.optimizer_name = opt_name

        if opt_name == 'SGD' : opt = optim.SGD(opt_params, weight_decay=wd, momentum=mom, nesterov=nesterov)
        
        elif opt_name == 'Adam': opt = optim.Adam(opt_params, weight_decay=wd)
            
        elif opt_name == 'RMSProp': opt = optim.RMSprop(opt_params, weight_decay=wd, momentum=mom)
        
        elif opt_name == 'Adagrad': opt = optim.Adagrad(opt_params, weight_decay=wd)
        else: raise ValueError("This optimizer not included, included ones are SGD, Adam, RMSProp and Adagrad") 
            
        print("optimizer set as : ",opt)
        
        self.optimizer = opt
        
        return        
        
    def get_data(self, data_transforms={}, num_workers=0):
        '''
         data_transforms : dictionary of data_transforms with keys 'train', 'val' and 'test'
        '''
        if data_transforms :
            datasets = {x: self.dataset_cls(x, transform=data_transforms[x]) for x in ['train', 'val']}
        else:
            datasets = {x: self.dataset_cls(x, transform=preprocess) for x in ['train', 'val']}
            
        self.dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=self.batch_size,
                                             shuffle=True, num_workers=num_workers, pin_memory=True)
              for x in ['train', 'val']}
        self.dataset_sizes = {x: datasets[x].len for x in ['train', 'val']}
        print("dataset_sizes",self.dataset_sizes)
        return
    
    def load_checkpoint(self):
        state = torch.load(self.out_dir+'/checkpoint.pth.tar')
        last_epoch = state['epoch']
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        best_model_wts = state['best_model_wts']
        if self.scheduler:
            self.scheduler.load_state_dict(state['scheduler'], self.model)
        min_loss = state['min_loss']
        return last_epoch, best_model_wts, min_loss


    def save_checkpoint(self, epoch, best_model_wts, min_loss):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'min_loss': min_loss,
                    'best_model_wts': best_model_wts                   
                 }

        if self.scheduler : state['scheduler'] = self.scheduler.state_dict()                    
        torch.save(state, self.out_dir+'/checkpoint.pth.tar')
    
    def save_experiment(self):
        experiment = self.title  + ',' + self.use_gpu + ',' + self.batch_size + ',' + self.dataset_cls + ',' + \
        self.dataset_sizes + ',' +self.out_dir + ',' + self.model + ',' + self.optimizer + ',' + self.scheduler + ',' + \
        self.criterion + ',' +self.time_elapsed + ',' + self.tta + ',' + self.eval_metric + ',' + self.metric_val
        print(experiment)            
        f = open('experiments.csv','a')
        f.write(experiment) 
        f.close()
    
    def plot_lrs(self):
        if self.scheduler.lr_history:
            fig, ax = plt.subplots(figsize=(20, 4))
            ax.plot([x[0] for x in self.scheduler.lr_history])
            ax.set_yscale("log")
            plt.show()
        
    def find_lr(self, init_value = 1e-8, final_value=10., beta = 0.98):
        '''
        Technique from the paper : Cyclical Learning Rates for Training Neural Networks
        Adapted from https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html#how-do-you-find-a-good-learning-rate
        and fastai library
        Over an epoch begin your SGD with a very low learning rate (like 1e−8) but 
        change it (by multiplying it by a certain factor for instance) at each mini-batch until
        it reaches a very high value (like 1 or 10). 
        Record the loss each time at each iteration and once you're finished, plot those losses against the learning rate.
        The loss decreases at the beginning, then it stops and it goes back increasing, usually extremely quickly. 
        That's because with very low learning rates, we get better and better, especially since we increase them. 
        Then comes a point where we reach a value that's too high. 
        '''
        trn_loader = self.dataloaders['train']
        num = len(trn_loader)-1

        use_gpu = self.use_gpu

        #how much to multiply our learning rate at each step. If we begin with a learning rate of lr(0)
        # and multiply it at each step by q then at the i-th step, our learning rate will be 
        # lr(i) = lr(0)×q(i) paranthesis indicating subscripts
        # q=(lr(N−1)/lr(0))** (1/N-1)

        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for data in trn_loader:
            batch_num += 1

            # get the inputs

            inputs, labels = data

            #Get the loss for this mini-batch of inputs/outputs
            if use_gpu:
                torch.cuda.synchronize()
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) *loss.data[0]
            smoothed_loss = avg_loss / (1 - beta**batch_num)

            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                print("loss exploding")
                #print(log_lrs)
                plot_lr_finder(log_lrs, losses)
                return 
            
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            #Do the SGD step
            loss.backward()
            self.optimizer.step()
            #Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr
        plot_lr_finder(log_lrs, losses)
        #print(log_lrs)
        return
    
    def _precompute_activ(self, net_name):
        trn_loader = self.dataloaders['train']
        use_gpu = self.use_gpu
   
        #feature_extract_lyrs = torch.nn.Sequential(*list(self.model.net.modules())[:-1])
        all_outputs = []
        all_labels = []

        feature_extract_lyrs = copy.deepcopy(torch.nn.Sequential(*list(self.model.net.children())[:self.model.feature_layers_num]))
        feature_extract_lyrs.train(False)
        feature_extract_lyrs = feature_extract_lyrs.cuda()
        for data in trn_loader:
            inputs, labels = data

            if use_gpu:
                torch.cuda.synchronize()
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            #pdb.set_trace()
            outputs = feature_extract_lyrs(inputs)
            all_outputs.append(outputs)
            all_labels.append(labels)
        #return torch.cat(all_outputs)       
        return torch.save({'outputs':all_outputs, 'labels':all_labels}, self.out_dir+'/precomputed.pth.tar')
    
    def _train_fc_layers(self, classifier_lyrs, num_epochs):
        precomputed  = torch.load(self.out_dir+'/precomputed.pth.tar')
        
        #Test
        #print("before model fc:",self.model.net.fc.state_dict()["weight"][0,0])
        #print("before last layers:",classifier_lyrs.fc.state_dict())
        #print("before last layers:",classifier_lyrs.fc.state_dict()["0.weight"][0,0])
        
        for layer in classifier_lyrs.children():
            print(layer)
            if isinstance(layer, torch.nn.Linear):
                fc = layer.state_dict()
        #print(fc["weight"].size())
        #print(fc["bias"].size())
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            classifier_lyrs.train(True)  # Set model to training mode
                
            running_loss = 0.0
            running_corrects = 0

            for inputs , labels  in zip(precomputed['outputs'], precomputed['labels']):

                # zero the parameter gradients
                self.optimizer.zero_grad()
                #print(inputs.shape)
                #print(labels.shape)
    
                # forward
                outputs = classifier_lyrs(inputs)
                #print(outputs.shape)
                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                # backward + optimize 
                
                loss.backward()
                self.optimizer.step()
              # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / self.dataset_sizes['train']
            epoch_acc = running_corrects / self.dataset_sizes['train']
    
            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
             

    # load model weights
        #count = 0
        #for idx1, model_layer in enumerate(self.model.net.children()):
            #if idx1 >= self.model.feature_layers_num:
                #for idx, layer in enumerate(classifier_lyrs.children()):
                    #if (count == idx) and (isinstance(layer, torch.nn.Linear)):
                        #fc = layer.state_dict()
                        #model_layer.load_state_dict({"weight":fc["0.weight"], "bias":fc["0.bias"]})
                        #model_layer.load_state_dict(copy.deepcopy(layer.state_dict()))
                #count += 1 
        fc = classifier_lyrs.fc.state_dict()       
        self.model.net.fc.load_state_dict({"weight":fc["0.weight"], "bias":fc["0.bias"]})
        #Test
        #print("after model fc:",self.model.net.fc.state_dict()["weight"][0,0]) 
        #print("after last layers:",classifier_lyrs.fc.state_dict()["0.weight"][0,0]) 
        #print("optim_parameters ", self.model.get_optim_params())
        
        return 
        
    
def plot_lr_finder(log_lrs, losses, n_skip=10, n_skip_end=5):
    '''
    Plots the loss function with respect to learning rate, in log scale. 
    TO DO : take plot parameters as optional arguments

    '''
    plt.ylabel("loss")
    plt.xlabel("learning rate (log scale)")
    plt.plot(log_lrs[n_skip:-(n_skip_end+1)], losses[n_skip:-(n_skip_end+1)])
    #plt.plot(log_lrs, losses)
    #plt.ylim(4,8)
    #plt.plot(log_lrs, losses)
    #plt.xscale('symlog')
    plt.show()
    return 



    
    
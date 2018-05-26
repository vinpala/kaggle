import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pdb
import h5py
import PIL
from PIL import Image


#Dataset class is used to provide an interface for accessing all the training or testing samples. In order to achieve this, 
#we have to implement two method, __getitem__ and __len__ so that each training sample (in image classification, a sample means an image  #plus its class label) can be accessed by its index.

class FurnitureDataset(Dataset):
    """iMaterialist Furniture dataset."""

    def __init__(self, prefix, source='local', transform=None):
        """
        Args:
            prefix (string): 'val'/'train'/'test
            source(string): Local or Cloud(IBM Cloud Object Storage)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.prefix = prefix
        self.source = source
        self.transform = transform
        
        #local
        if prefix == 'train':
            dataset_name = 'train_1'
        elif prefix == 'val':
            dataset_name = 'validation_last'
        else:
            pass
            
        self.images = h5py.File('data/Sample/{}_images.h5'.format(dataset_name), 'r')
        self.labels = h5py.File('data/Sample/{}_labels.h5'.format(dataset_name), 'r')
        self.len = self.labels['{}_labels'.format(dataset_name)].shape[0]
        self.classes = set(self.labels['{}_labels'.format(dataset_name)][:])

    def __len__(self):
        
        return self.len

    def __getitem__(self, idx):

        image, label = self.get_data_from_local(idx)
        #else: 
        #self.data = get_data_from_cloud(prefix, idx)
        image = Image.fromarray(np.uint8(image))
        if self.transform: image = self.transform(image)
        
        return image, label
    
    def get_data_from_COS(self, prefix, idx):
        pass

    def get_data_from_local(self, idx):
        """
        Get the image and label for the idx from HDF5 files
        """
        if self.prefix == 'train':
            dataset_name = 'train_1'
        elif self.prefix == 'val':
            dataset_name = 'validation_last'
        else:
            pass
        image = self.images['{}_images'.format(dataset_name)][idx]
        label = self.labels['{}_labels'.format(dataset_name)][idx]
    
        return image, label
    
#def get_image_as_array(image):
#    return np.asarray(image)
    
def onehot(label, num_classes):

    """
    one hot encode labels

    """
    onehot = np.zeros((num_classes), dtype=np.float32)
    onehot[label] = 1
    return onehot
## Additional transforms for PyTorch data augmentation
## Transforms in OpenCv are way faster than PIL

import random

import torch

import numpy as np

import cv2 #open-CV

import PIL.ImageEnhance as ie

import PIL.Image as im

import torchvision.transforms.functional as F

from torchvision import transforms

from config import IMAGE_SIZE

"""Statistics pertaining to image data from image net. mean and std of the images of each color channel"""
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


#https://github.com/skrypka/imaterialist-furniture-2018/blob/master/augmentation.py
class HorizontalFlip(object):

    """Horizontally flip the given PIL Image."""
    def __call__(self, img):

        """

        Args:

            img (PIL Image): Image to be flipped.



        Returns:

            PIL Image: Flipped image.

        """

        return F.hflip(img)
    

#https://github.com/mratsim/Amazon-Forest-Computer-Vision/blob/master/src/p_data_augmentation.py#L181
class RandomRotate(object):

    """Randomly rotate the given PIL.Image with a probability of 1/6 90°,

                                                                 1/6 180°,

                                                                 1/6 270°,

                                                                 1/2 as is

    """

    def __call__(self, img):

        dispatcher = {

            0: img,

            1: img,

            2: img,            

            3: img.transpose(im.ROTATE_90),

            4: img.transpose(im.ROTATE_180),

            5: img.transpose(im.ROTATE_270)

        }

    

        return dispatcher[random.randint(0,5)] #randint is inclusive

class Pad(object):
    '''
    Add a reflection padding to image(best practice ref: fastai library)
    '''
    def __call__(self, img, pad=4, mode=cv2.BORDER_REFLECT):

        """

        Args:

            img (PIL Image): Image to be padded
            pad :  size of padding on top, bottom, left and right
            mode :type of cv2 padding modes(default : reflection padding)

        Returns:

            PIL Image: padded image.

        """
        img = ToCv2(img) 
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, mode)
        return ToPIL(img)

def ToCv2(PILimg):
    '''
    Convert PIL image to CV2
    '''
    open_cv_image = np.array(PILimg.convert('RGB'))
# Convert RGB to BGR 
    #open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def ToPIL(Cv2img):
    '''
    Convert Cv2 image to PIL
    '''
    img = cv2.cvtColor(Cv2img, cv2.COLOR_BGR2RGB)
    return im.fromarray(img)

def ToNumpy(torch_image):
    """
    To convert to numpy image 
    
    Torch stores data in a channel-first mode while numpy and PIL work with channel-last.
    torch image: C X H X W
    numpy image: H x W x C
    """
    return torch_image.numpy().transpose((1, 2, 0))


def denormalize(numpy_image):
    
    """
    Denormalize to display (RGB)
    """
    # this is how normalization is done input[channel] = (input[channel] - mean[channel]) / std[channel]
    # to undo this input[channel] = (input[channel] * std[channel]) + mean[channel]
    # mean=imagenet_stats[0], std=imagenet_stats[1]
    new_image = np.zeros(numpy_image.shape)
    
    for channel, mean, std in zip(range(3), imagenet_stats[0], imagenet_stats[1]):       
        new_image[channel] = (numpy_image[channel] * std) + mean
    return new_image

#**************************************************************************************************************
    
normalize = transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])

preprocess = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    normalize

])

preprocess_hflip = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    
    RandomRotate(),

    HorizontalFlip(),

    transforms.ToTensor(),
    
    normalize

])


preprocess_with_augmentation = transforms.Compose([

    #transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    
    Pad(),#reflection padding

    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.ColorJitter(brightness=0.3,

                           contrast=0.3,

                           saturation=0.3),

    transforms.ToTensor(),

    normalize

])

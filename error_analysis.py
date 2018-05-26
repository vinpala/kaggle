import config
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import transforms
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import dataset
from transforms import ToNumpy, denormalize


def analyze_results(classifier, test_dataset, probs):
    print("Analyzing results")
    y = test_dataset.data["labels"]
    preds = np.argmax(probs, axis=1)
    #stacked = np.stack((y, preds, probs), axis=-1).tolist()
    #columns = ['label', 'pred'] 
    #columns = columns.append(list(map(str, range(classes)))) 
    #df_stacked = pd.DataFrame(stacked, columns=columns)
    cm = confusion_matrix(y, preds)
    if NUM_CLASSES <= 10 :
        plot_confusion_matrix(cm, classes)
    most_correct, most_incorrect = analyze_confusion(cm)    
    plot_with_title(test_dataset, y , preds, classes=most_correct, sup_title="Most correct")
    plot_with_title(test_dataset, y, preds, classes=most_incorrect, sup_title="Most incorrect")
    plot_most_per_cls(y, pred, prob, is_correct=True, sup_title="Most correct per class" )
    plot_most_per_cls(y, pred, prob, is_correct=False, sup_title="Most incorrect per class")
    
def plot_most_per_cls(y, pred, prob, is_correct=True, sup_title=None):
    if is_correct: mult = 1
    else:
        mult = -1   
    
    mask = (y==preds) == is_correct
    idxs = np.where(mask)
    prob_select = probs[idxs] 
    titles = []
    imgs = []
    for idx in range(prob_select.shape[1]):
        imgs = np.argsort(mult * probs_select[:,idx])[0:5]
        titles = probs_select[imgs:,idx]
        plot_with_title(test_dataset, image_idxs=imgs, sup_title=sup_title, titles=titles)   
    
def plots(imgs, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(imgs)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        img_denorm = denormalize(ToNumpy(imgs[i]))
        img_denorm = (1/(2*2.25)) * img_denorm + 0.5
# https://stackoverflow.com/questions/47318871/valueerror-floating-point-image-rgb-values-must-be-in-the-0-1-range-while-usi
#you must re-scale them to the range 0.0 - 1.0.
#You want to retain visibility of negative vs positive, but that you let 0.5 be your new "neutral" point. 
#Scale such that current 0.0 values map to 0.5, and your most extreme value (largest magnitude) scale to 0.0 (if negative) or 1.0 (if positive).
#It looks like the values are in the range -2.25 to +2.0. I suggest a rescaling new = (1/(2*2.25)) * old + 0.5
        plt.imshow(img_denorm)

def plot_with_title(test_dataset, y, preds, classes=None, image_idxs=None, sup_title=None, titles=None):
    if len(classes) == 0 & len(image_idxs) == 0:
        raise ValueError("pass classes or image_idx")       
    if len(classes) > 0:
        for cls in classes:
            image_idxs = np.where(y == preds & y == cls)[0:5] 
    imgs = [dataset.get_image_as_array(test_dataset, x) for x in image_idxs]
    print(sup_title)
    return plots(imgs, rows=1, titles=titles, figsize=(16,8))

def plot_confusion_matrix(cm, classes):
    df_cm = pd.DataFrame(array, classes, classes)
    plt.figure(figsize = (10,7))
    plt.title("Confusion Matrix")
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    
def analyze_confusion(cm):
    totals = np.sum(cm, axis=1)
    print("Classes with the most no. of samples in the test set : ",np.argsort(totals)[::-1][0:20])
    print("Number of samples for these                          : ",sorted(totals)[::-1][0:20])
    print("Classes with the least no.of samples in the test set : ",np.argsort(totals)[0:20])
    print("Number of samples for these                          : ",sorted(totals)[0:20])
    cm = (cm/totals[:,None]) *100 #express as percentages
    sort_dia = np.argsort(cm.diagonal())
    most_correct = sort_dia[::-1][0:20]
    most_incorrect = sort_dia[0:20]
    print("Classes with most predictions correct :", most_correct)
    print("Classes with most predictions incorrect :", most_incorrect)
    class_confused =[]
    for i in range(cm.shape[0]):
        if i == np.argsort(cm[i,:])[-1]:
            class_confused.append(np.argsort(cm[i,:])[-2])
        else: 
            class_confused.append(np.argsort(cm[i,:])[-1])  
    for i in range(cm.shape[0]):
        print("Class {1} confused with class {2} : {3} nos".format(i, class_confused[i], cm[i, class_confused[i]]))
    return most_correct, most_incorrect
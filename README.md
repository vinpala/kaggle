# Imaterialist Kaggle Challenge 2018(ongoing competition)
This is a work-in-progress. Nevertheless, posting whatever I have at the moment.

## My Approach

I uploaded the complete competition data to IBM cloud storage (check Load_COS notebook) and intend to run the models on the complete data from IBM Watson Studio.(There are quite a lot of image urls that are not found right now, but somebody on the Kaggle forum was kind enough to upload almost-complete dataset to his Google drive and share the link. I guess for training the missing urls shouldnt be an issue but I intend to use his test images).

I downloaded a small subset of training data(5 GB) and validation data(2 GB) to my laptop(with NVIDIA GEFORCE-GTX 1050 Ti) to visualize and possibly run models on this subset to check for errors and also set hyperparameters to overfit models on this sample. The plan is to later train the entire dataset (now on IBM cloud) on these models from iBM Watson Studio.

I am trying transfer learning with RESNET, RESNEXT, DENSENET etc and using the best practises mentioned in the fastai 2018 MOOC for image classification. But I am cool enough to resist the temptation of using their fastai library and I am building <b> my own library </b>
on top of pytorch. Here are the techniques i am or will be trying -
<br>
#### 1) Test Time Augmentation
#### 2) Cosine Annealing with restarts learning rate
#### 3) One learning rate for superconvergence
#### 4) Learning rate finder
#### 5) Differential learning rate
#### 6) Using weighted loss function to tackle class imbalance
#### 7) Ensembling method used by previous kaggle winners

I am not in a hurry to meet the competition deadline because i consider this a great learrning experience.

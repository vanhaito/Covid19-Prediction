# X-ray Diagnosis for Covid-19

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Framework Used](#framework-used)
  * [Data Collection](#data-collection)
  * [Deep Learning Model](#deep-learning-model)
  * [Result](#result)


## Demo
![](https://github.com/vanhaito/Flask-Covid19-Prediction-WebApp/blob/main/demo.png)
## Overview
 - This is a university group project about Artificial Intelligence. The CNN model takes an X-ray image and predicts if the image belongs to 1 of 3 classes with a probability: Normal, Covid, or Viral pneumonia. 
## Framework Used
 - Keras
 - Flask
## Data Collection
- The dataset used takes 3000 images of the 3 classes: normal, COVID, and viral penumonia, respectively.
  - Train set contains 64 %
  - Validation set contains 16%
  - Test scontains 20%
- Source: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
## Deep Learning Model
For this problem, a convolutional neural network was used to train the model. The model receives a raw image, then proceeds to clean it. After that, the cleaned image is fed into the CNN model, which proceeds to train and, in the end, classify the image with a probability for each class.
![](https://github.com/vanhaito/Flask-Covid19-Prediction-WebApp/blob/main/model.png)

## Result
The result was on the test set with 94.5% accuracy.

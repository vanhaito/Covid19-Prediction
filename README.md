# X-ray Diagnosis for Covid-19

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Framework Used](#framework-used)
  * [Data Collection](#data-collection)
  * [Deeplearning Model](#deeplearning-model)
  * [Result](#result)


## Demo
![](https://github.com/vanhaito/Flask-Covid19-Prediction-WebApp/blob/main/demo.png)
## Overview
 - This is a university group project about Artificial Intelligence. The CNN model takes an X-ray image and predicts if the image belongs to 1 of 3 classes with a probability: Normal, Covid, or Viral pneumonia. 
## Framework Used
 - Keras
 - Flask
## Data Collection
The dataset used takes 3000 images of the 3 classes: normal, COVID, and viral penumonia, respectively, from source: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
## Deeplearning Model
In this problem, a convolutional neural network was used to train the model. The model receives a raw image, then proceeds to clean it. After that, the cleaned image is fed into the CNN model, that proceeds to train and in the end classify the image.
![](https://github.com/vanhaito/Flask-Covid19-Prediction-WebApp/blob/main/model.png)

## Result

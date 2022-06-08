# Regression with feed-forward neural network

## Introduction

This repository contains an experimental simulator for 2D regression using a feed-forward neural network. It can be used to create, train and evaluate models on a given set of data created from y(x) = (x+0.8)*(x-0.2)*(x-0.3)*(x-0.6) in interval [-1,+1]. 

## Usage

- Open <a href="http://public.beuth-hochschule.de/~s85393/deep-learning/regression-with-ffnn/" target="_blank" rel="noopener noreferrer">regression with ffnn</a> in web browser
- Select number of samples to be used for training and testing
- Set network paramaters or use presets available and click `Create Model`
- Set number of epochs or use presets and click `Train Model`
- When training is finished click `Test Model` to test model on samples selected
- If you want to test the model on a different number of samples change number of samples accordingly and click `Test Model` again
- Try out different model and training parameters and see results

## Technical Documentation

This section provides an overview about the technical setup of the project. Project depedencies are listed and crucial components, functions and attributes are explained.

### Local Usage 

- Clone repository
- Open `index.html` in web browser

### Dependencies

This section contains a list of all project dependencies. 
Name | Description | Reference |
--- | --- | --- | 
| Bootstrap | Bootstrap is a CSS library including JavsScript to style user interfaces and to create user interaction. Within this project Bootstrap v5.2 is used to style user interface and components in a convenient way, and also to add tooltips for context sensitive help | https://getbootstrap.com/ |
| Tensorflow.js | Tensorflow.js makes deep learning available in the browser or within Node.js. In this project it used for model creating, persistence, training and evaluation | https://www.tensorflow.org/js |
| Tensorflow.js VIS| Tensorflow.js VIS is a library to create UI helpers to be used with tensorflow.js. Within this project it is used for visualization of data, model, training and evaluation | https://js.tensorflow.org/api_vis/latest/|


## Experiments & Results

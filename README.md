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

### Saving & Loading Models
When a model has been created and trained the model topology and its weights can be saved to the browsers local storage. To save and load a model:
- Create and train model
- Click `Save Model` in the model section
- Click `Load Model` in the model section

This function can be helpful if you want save a certain model for later and want to try out different model topologies. If you save a model which has not been trained, only the model topology is saved.

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

### Functions
#### Data Sampling

Name | Description |
--- | --- | 
getUniformDistributedRandomNumber(min, max) | Creates uniform distributed number in interval |
addGaussianNoise(y, mean, variance) | Adds gaussian noise to y value |
calcSum(array) | Helper function to calculate sum for array of numbers |
calcMean(array) | Helper function to calculate mean for array of numbers |
line 73 | Loop to create sample data n from N from given function. Creates x and y values. Stores results in sampled array  |
line 87| Loop to add gaussian noise to each y of n from N |

#### Model Creation, Training und Evaluation

Name | Description |
--- | --- | 
create() | Helper function to call createModel() on user interaction and to provide user feedback and visualization |
train() | Helper function to create tensor data and to call trainModel() on user interaction and to provide user feedback and visualization |
test() | Helper function to call testModel() on user interaction |
save () | Helper function to save model topology and weights to browser local storage and to provide user feedback |
load() | Helper function to load model topology and weights from browser local storage to visualize loaded model |
createModel() | Creates model based on given parameters hidden layers, neurons per layer and activation function  |
convertToTensor(data) | Creates tensor data from sampled data used to train model |
trainModel() | Used to compile and fit model on input data provided |
testModel() | Used to evaluate fitted model on given data |

# MNIST Dataset Example
This project demonstrates how to perform image classification using the MNIST dataset with machine learning algorithms, focusing specifically on implementing a Convolutional Neural Network (CNN). The MNIST dataset consists of 70,000 handwritten digits (0-9) and is a good example of a supervised learning problem problem in image recognition tasks. A CNN is used in ML when the input data consists of images or spatial data, and the goal is to automatically learn patterns such as edges, textures, and shapes. 

The goal is to showcase fundamental techniques in image classification and machine learning, highlighting how models can learn from image data to recognize and classify visual patterns with high accuracy. 
# Table Of Contents
- [Implementation](#implementation)
- [Requirements](#requirments)
- [How to Use](#how-to-use)
- [Error Handling](#error-handling)
- [References](#references)
# Implementation
The models implementation consists of an input of the MNIST dataset that involves a collection of 70,000 handwritten digits (0-9), with each image being 28x28 pixels. The purpose of training this model is to teach the neural network to identify and correctly classify handwritten digits based on pixel intensity patterns. The model is trained using 25 epochs, meaning it is passed through 25 times, with a validation split of 0.2 to prevent overfitting. This dataset is used widely for educational puroposes in deep learning and computer vision research. 

Performance metrics are used within machine learning to validate the models performance. In this implementation, the training loss and accuracy was looked at along with the validation loss and accuracy. These metrics used in conjuction tell us how effectively the model is learning over time, whether it is generalizing well to unseen data, and if issues like overfitting or underfitting are present. These figures along with their interpretation can be found in the [media](#media) folder. 
# Requirments 
This project requires tensorflow, keras, matplotlib and scikit-learn. It was developed using a Python environment through VSCode.

Use 'pip install -r requirements.txt' to install the following dependencies:

```
tensorflow==2.20.0
matplotlib==3.10.6
keras==3.11.3
scikit-learn==1.7.1
```
# How to Use
To utilize this code, a Python environment should be installed onto the MNIST.py file onto your computer into a folder. Then, open that folder/file on VS Code.

Another way to open this file is to clone the repository within Github. To do this, you press the Code button on Github and copy the HTTPS URL. When you open VS Code, you will be prompted to make a selection. One of the options is to clone a repository, and if that is pressed, you can paste the URL. This will open the repository into Github.
# Error Handling 
This project does not have any error handling.
# References 
- [1]GeeksforGeeks, “MNIST Dataset : Practical Applications Using Keras and PyTorch,” GeeksforGeeks, May 2024. https://www.geeksforgeeks.org/machine-learning/mnist-dataset/
- [2]A. Khan, “A Beginner’s Guide to Deep Learning with MNIST Dataset,” Medium, Apr. 16, 2024. https://medium.com/@azimkhan8018/a-beginners-guide-to-deep-learning-with-mnist-dataset-0894f7183344
‌

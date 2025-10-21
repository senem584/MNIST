# MNIST_Dataset_Example
This project demostrates the MNIST dataset which contains a large collection of handwritten digits used to train a neural network while utilizing TensorFlow + Keras. The goal is for demostration of image classification. 
# Table Of Contents
- [Implementation](#implementation)
- [Requirements](#requirments)
- [How to Use](#how-to-use)
- [Error Handling](#error-handling)
- [References](#references)
# Implementation
The models implementation consists of an input of the MNIST dataset in which it involves a collection of 70,000 handwritten digits (0-9), with each image being 28x28 pixels. The model is trained using 25 EPOCHs in which it is passed through 25 times, with a validation split of 0.2 to prevent overfitting and validates on a specific portion of the data. 
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

# MNIST Dataset Example
This project demonstrates how to perform image classification using the MNIST dataset with machine learning algorithms, focusing specifically on implementing a Convolutional Neural Network (CNN). The MNIST dataset consists of 70,000 handwritten digits (0-9) and is a good example of a supervised learning problem problem in image recognition tasks. A CNN is used in ML when the input data consists of images or spatial data, and the goal is to automatically learn patterns such as edges, textures, and shapes. 

The goal is to showcase fundamental techniques in image classification and machine learning, highlighting how models can learn from image data to recognize and classify visual patterns with high accuracy. The code is implemented in two styles: a regular procedural version and an object-oriented (OOP) version for better modularity and reusability. Both are shown in the source folder.
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
absl-py==2.3.1
astunparse==1.6.3
certifi==2025.10.5
charset-normalizer==3.4.4
contourpy==1.3.3
cycler==0.12.1
flatbuffers==25.9.23
fonttools==4.60.1
gast==0.6.0
google-pasta==0.2.0
grpcio==1.75.1
h5py==3.15.1
idna==3.11
joblib==1.5.2
keras==3.11.3
kiwisolver==1.4.9
libclang==18.1.1
Markdown==3.9
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.7
mdurl==0.1.2
ml_dtypes==0.5.3
namex==0.1.0
numpy==2.3.4
opt_einsum==3.4.0
optree==0.17.0
packaging==25.0
pillow==12.0.0
protobuf==6.33.0
Pygments==2.19.2
pyparsing==3.2.5
python-dateutil==2.9.0.post0
requests==2.32.5
rich==14.2.0
scikit-learn==1.7.2
scipy==1.16.2
setuptools==80.9.0
six==1.17.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorflow==2.20.0
termcolor==3.1.0
threadpoolctl==3.6.0
typing_extensions==4.15.0
urllib3==2.5.0
Werkzeug==3.1.3
wheel==0.45.1
wrapt==2.0.0
```
# How to Use
Clone the repository
- On GitHub, click the Code button and copy the HTTPS URL.
- In VS Code, choose Clone Repository, then paste the URL.

Run the file
- Locate the file named MNIST.py.

Run using:
- Make sure your enviornment is active.
- python MNIST.py

# References 
- [1]GeeksforGeeks, “MNIST Dataset : Practical Applications Using Keras and PyTorch,” GeeksforGeeks, May 2024. https://www.geeksforgeeks.org/machine-learning/mnist-dataset/
- [2]A. Khan, “A Beginner’s Guide to Deep Learning with MNIST Dataset,” Medium, Apr. 16, 2024. https://medium.com/@azimkhan8018/a-beginners-guide-to-deep-learning-with-mnist-dataset-0894f7183344
‌

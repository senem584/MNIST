import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class MNISTTrainer:
    def __init__(self, epochs=25, val_split=0.2, hidden1=128, hidden2=32, num_classes=10):
        # basic hyperparameters you can tweak
        self.epochs = epochs
        self.val_split = val_split
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_classes = num_classes

        # placeholders for data/model/history
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.history = None

    def load_data(self):
        # uploading the MNIST dataset using the keras API
        # using load_data() splits the dataset into 2 parts: x_train and y_train
        # x_train (images) and y_train (labels) are images of handwritten digits and their labels
        # x_test (images) and y_test (labels) are images and labels of handwritten digits for testing the model
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

        # checking the dataset to make sure i understand
        # 28x28 pixels in the X sets
        # 60k samples in training and 10k samples in testing
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    def preprocess(self):
        # data preprocessing
        # we are training and testing with images, so it is important to normalize the pixel values
        # pixel values range from 0-255, so we normalize to 0-1
        # normalizing allows our model to learn on a smaller scale, so it does not weigh large pixel values as much larger
        # x_train and x_test are the only sets with images (the y sets are labels, so they are in an array format)
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

    def build_model(self):
        # building the neural network model
        # Sequential() is used for building neural networks by stacking layers linearly and sequentially
        # we can use Sequential() because we have one input and one output
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))  # converting the 2D image into a 1D array

        # Dense layer = a layer with neurons that are connected to the outputs of the previous layer
        # activation function (we are using ReLU) decides what can pass forward
        # ReLU basically says if it's positive, keep it. if it is negative, don't. this helps models learn patterns
        model.add(Dense(self.hidden1, activation='relu'))  # 128 neurons looks at input data and learns to detect a pattern
        model.add(Dense(self.hidden2, activation='relu'))  # compressing features to the most important info

        # MNIST has 10 classes (digits 0-9), so we use 10 outputs with softmax
        model.add(Dense(self.num_classes, activation='softmax'))  # producing class probabilities

        model.summary()
        self.model = model

    def compile_and_train(self):
        # compiling and training the model
        # epochs = iterations: 1 epoch = 1 complete pass through training dataset
        # through each epoch, weights and biases are adjusted
        # validation_split: prevents overfitting. validates performance on a portion of data
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy']
        )
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            validation_split=self.val_split
        )

    def evaluate(self):
        # model evaluation
        y_prob = self.model.predict(self.X_test)
        y_pred = y_prob.argmax(axis=1)
        acc = accuracy_score(self.y_test, y_pred)
        print("Test accuracy (sklearn):", acc)
        return acc

    def plot_metrics(self):
        # model loss
        plt.figure()
        plt.plot(self.history.history['loss'], label='train loss')
        plt.plot(self.history.history['val_loss'], label='val loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        # model accuracy
        plt.figure()
        plt.plot(self.history.history['accuracy'], label='train acc')
        plt.plot(self.history.history['val_accuracy'], label='val acc')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        self.load_data()
        self.preprocess()
        self.build_model()
        self.compile_and_train()
        self.evaluate()
        self.plot_metrics()


if __name__ == "__main__":
    trainer = MNISTTrainer(
        epochs=25,
        val_split=0.2,
        hidden1=128,
        hidden2=32,
        num_classes=10  # digits 0-9
    )
    trainer.run()

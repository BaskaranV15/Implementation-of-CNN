# Implementation-of-CNN

## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset
The goal of this project is to develop a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. Handwritten digit classification is a fundamental task in image processing and machine learning, with various applications such as postal code recognition, bank check processing, and optical character recognition systems.

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9), totaling 60,000 training images and 10,000 test images. The challenge is to train a deep learning model that accurately classifies the images into the corresponding digits.

![image](https://github.com/user-attachments/assets/0300b590-2d11-4fc0-a1d1-49abcd0803a1)

## Neural Network Model

Include the neural network model diagram.(http://alexlenail.me/NN-SVG/index.html)
![Screenshot 2024-09-30 103054](https://github.com/user-attachments/assets/33ec9dde-220f-43f4-b0f9-23ca34bae7e8)


## DESIGN STEPS

### STEP 1:
Import the necessary libraries and Load the data set.

### STEP 2:
Reshape and normalize the data.
### STEP 3:
In the EarlyStoppingCallback change define the on_epoch_end funtion and define the necessary condition for accuracy.
### STEP 4:
Build model Architecture and fit to the model.train the model 
## PROGRAM

### Name:
### Register Number:

Name: V.BASKARAN
Register Number: 212222230020
### Loading and Inspecting Data And Normalizing and Reshaping
``` python
import os
import base64
import numpy as np
import tensorflow as tf


# Append data/mnist.npz to the previous path to get the full path
data_path = "mnist.npz.zip"

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")
def reshape_and_normalize(images):
    """Reshapes the array of images and normalizes pixel values.

    Args:
        images (numpy.ndarray): The images encoded as numpy arrays

    Returns:
        numpy.ndarray: The reshaped and normalized images.
    """

    ### START CODE HERE ###

    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = images[..., np.newaxis]

    # Normalize pixel values
    images = images / 255.0

    ### END CODE HERE ###

    return images
# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)

```
### Early Stopping Callback
``` python
import tensorflow as tf

class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Check if the accuracy is greater or equal to 0.995
        if logs.get('accuracy') >= 0.995:
            # Stop training once the above condition is met
            self.model.stop_training = True
            print("\nReached 99.5% accuracy, so cancelling training!")

```
### Defining the Convolutional Neural Network (CNN) Model
``` python

def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """


    # Define the model
    model = tf.keras.models.Sequential([
        # Convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # Max pooling layer
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # Flatten layer to convert 2D outputs to 1D
        tf.keras.layers.Flatten(),
        # Dense layer with 64 units and ReLU activation
        tf.keras.layers.Dense(64, activation='relu'),
        # Output layer with 10 units (one for each class) and softmax activation
        tf.keras.layers.Dense(10, activation='softmax')
    ])

     # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

```
### Training the Model
```python
model = convolutional_model()
training_history = model.fit(training_images, training_labels, epochs=15, callbacks=[EarlyStoppingCallback()])
```

### Prediction
```python
import numpy as np

# Reshape 'test' to add the channel dimension
test = test.reshape(1, 28, 28, 1)

# Now, try making predictions again
predictions = model.predict(test)
print(predictions)
predicted_classes = np.argmax(predictions, axis=1)

print(predicted_classes)
```
## OUTPUT

### Reshape and Normalize output

![image](https://github.com/user-attachments/assets/eae2bfbf-e4fe-4bb7-9d8f-59350ebd37ba)


### Training the model output

![image](https://github.com/user-attachments/assets/b09881d0-83a3-43c3-9b69-b99cbfec1cc1)

### Predicton output

![image](https://github.com/user-attachments/assets/c2470835-9074-4cf9-a892-fdfec0ff2072)


## RESULT
Thus, A convolutional deep neural network for digit classification is successfully executed.

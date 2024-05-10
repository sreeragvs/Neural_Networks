import os
import matplotlib.pyplot  # Matplotlib is imported but not used
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Setting TensorFlow log level to suppress warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Loading and preprocessing the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0  # Normalizing training data
x_test = x_test.astype("float32") / 255.0  # Normalizing test data
w
# Building the model
model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))  # Fixing the input shape
model.add(
    layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, activation='tanh')  # Bidirectional LSTM layer
    )
)

model.add(
    layers.Bidirectional(
        layers.GRU(256, activation='tanh')  # Bidirectional GRU layer
    )
)

model.add(layers.Dense(10))  # Output layer with 10 units (for 10 classes)

print(model.summary())  # Printing model summary

# Compiling the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Sparse categorical cross-entropy loss
    optimizer=keras.optimizers.Adam(lr=0.001),  # Adam optimizer with learning rate 0.001
    metrics=["accuracy"]
)

# Training the model
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)

# Evaluating the model
model.evaluate(x_test, y_test, batch_size=64, verbose=2)

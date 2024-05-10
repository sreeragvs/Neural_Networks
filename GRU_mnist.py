import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Define the model
model = keras.Sequential()

# Input layer with variable length sequences of 28 features
model.add(keras.Input(shape=(None, 28)))

# GRU layer with 256 units, returns sequences, and tanh activation
model.add(layers.GRU(256, return_sequences=True, activation='tanh'))

# Second GRU layer with 256 units and tanh activation
model.add(layers.GRU(256, activation='tanh'))

# Output layer with 10 units for 10 classes (digits)
model.add(layers.Dense(10))

# Print model summary
print(model.summary())  # corrected from print(model.summary)

# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)

# Evaluate the model on test data
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
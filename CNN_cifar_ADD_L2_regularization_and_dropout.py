import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Define the model architecture
def my_model():
    inputs = keras.Input(shape=(32,32,3))
    x = layers.Conv2D(
        32, 3,padding='same', kernel_regularizer=regularizers.l2(0.01),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        64, 5, padding='same', kernel_regularizer=regularizers.l2(0.01),
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(
        128, 3, padding='same', kernel_regularizer=regularizers.l2(0.01),
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)  # Flatten layer to convert 2D feature maps to 1D feature vectors
    x = layers.Dense(
        64, activation='relu', kernel_regularizer=regularizers.l2(0.01),
    )(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
# Create the model
model = my_model()
# Print model summary
print(model.summary())

# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Sparse categorical crossentropy loss function
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),  # Adam optimizer with learning rate 3e-4
    metrics=["accuracy"],  # Metric to monitor during training
)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=150, verbose=2)  # Train the model for 10 epochs with batch size 64

# Evaluate the model on test data
model.evaluate(x_test, y_test, batch_size=64, verbose=2)  # Evaluate the model performance on test data

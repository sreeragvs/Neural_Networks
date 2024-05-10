import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Define the model architecture
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),  # Input layer with shape (32,32,3)
        layers.Conv2D(32, 3, padding='valid', activation='relu'),  # Convolutional layer with 32 filters, kernel size 3x3, relu activation
        layers.MaxPooling2D(pool_size=(2, 2)),  # Max pooling layer with pool size 2x2
        layers.Conv2D(64, 3, activation='relu'),  # Convolutional layer with 64 filters, kernel size 3x3, relu activation
        layers.MaxPooling2D(),  # Max pooling layer with default pool size 2x2
        layers.Conv2D(128, 3, activation='relu'),  # Convolutional layer with 128 filters, kernel size 3x3, relu activation
        layers.Flatten(),  # Flatten layer to convert 2D feature maps to 1D feature vectors
        layers.Dense(64, activation='relu'),  # Fully connected layer with 64 units and relu activation
        layers.Dense(10),  # Output layer with 10 units (for 10 classes), no activation specified (for logits)
    ]
)

# Print model summary
print(model.summary())

# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Sparse categorical crossentropy loss function
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),  # Adam optimizer with learning rate 3e-4
    metrics=["accuracy"],  # Metric to monitor during training
)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)  # Train the model for 10 epochs with batch size 64

# Evaluate the model on test data
model.evaluate(x_test, y_test, batch_size=64, verbose=2)  # Evaluate the model performance on test data

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
mnist = keras.datasets.mnist
(X_train_full, y_train_full),(X_test,y_test) = mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255, X_train_full[5000:]/255
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test/255
class_names = ["0","1","2","3","4","5","6","7","8","9"] #define the class names.vsavm;m
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation = "relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation = "relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation = "softmax")])
print(model.summary())
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])#Compailing the Model( with some hyper parameters)
#now we fit the model with the data.
history = model.fit(X_train, y_train, epochs=30, validation_data = (X_valid, y_valid), batch_size=32) #also possible to use validation_split =0.1

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(15,8))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.evaluate(X_test, y_test)
#PREDICTIONS
#model.predict(X_test)

y_prob = model.predict(X_test)
y_classes = y_prob.argmax(axis=-1) #argmax gives the corresponding number
print(y_classes)
#view confusion matrix graphically
confusion_matrix = tf.math.confusion_matrix(y_test, y_classes)
import seaborn as sb

# ax = plt.figure(figsize=(8, 6))
fig = sb.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Greens')  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
fig.set_xlabel('Predicted labels')
fig.set_ylabel('True labels')
fig.set_title('Confusion Matrix')
fig.xaxis.set_ticklabels(class_names)
fig.yaxis.set_ticklabels(class_names)
fig.figure.set_size_inches(10, 10)


plt.show()
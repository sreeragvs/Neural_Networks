import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
mnist = keras.datasets.mnist
(x_train_full, y_train_full), (x_test, y_test) =mnist.load_data()
print(x_train_full.shape)
#print(x_test.shape)
#print(x_train_full[0])
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
a = 0

for i in range(3):
    for j in range(3):
        axes[i, j].imshow(x_train_full[a], cmap=plt.get_cmap('gray'))
        a = a + 1

#plt.show()
#normalizing the values(pixel values goes from 0 to 255) we are trying to put all the values from a scale of 0 to 1.
#we are also creating a validation set, for some internal testing for generalisation or smthg.
x_valid, x_train = x_train_full[:5000] / 255, x_train_full[5000:]/255
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test/255
class_names = ["0","1","2","3","4","5","6","7","8","9"] #define the class names.
#print(class_names[y_train[2]])
#plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
#plt.show()

#Neural network.
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28])) # this represent the input, we are sending 28x28 values, Flatten is used for sending all 784 values in single line.
model.add(keras.layers.Dense(300, activation = "relu")) #hidden layers, Dense means every layers are connected each other, we can also use Sparse which means all layers are not connected
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))

#Sigmoid: probabilities produced by a Sigmoid are independent.
#Softmax: the outputs are interrelated. The sum of all outputs are 1.

print(model.summary())
print(model.layers)
#Compailing the Model( with some hyper parameters)
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]) #Use this crossentropy loss function when there are two or more label classes.

#now we fit the model with the data.
history = model.fit(x_train, y_train, epochs=30, validation_data = (x_valid, y_valid), batch_size=32) #also possible to use validation_split =0.1

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(15,8))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.evaluate(x_test, y_test)
#PREDICTIONS
#model.predict(X_test)

y_prob = model.predict(x_test)
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
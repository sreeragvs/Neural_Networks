import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = tf.matmul(x,y)
print(z)
m = x@y
print(m)
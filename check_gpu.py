import tensorflow as tf

# Get a list of all available physical devices
devices = tf.config.list_physical_devices()

# Check if any GPUs are available
gpus = [device.name for device in devices if 'GPU' in device.device_type]

if gpus:
    print("TensorFlow version supports GPU acceleration")
    print("GPU device(s):", gpus)
else:
    print("TensorFlow version supports only CPU")
# Get the TensorFlow version
print("TensorFlow version:", tf.__version__)
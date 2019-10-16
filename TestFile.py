import gzip
import matplotlib.pyplot as plt
import numpy as np
import keras as kr

# ----------------------- Testing MNIST Reading --------------------------------
# Reading bytes from data files
with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    file_content = f.read()
file_content[0:4]

# Reading images from data files
l = file_content[16:800]
# print(type(l)) - check if type is bytes
image = ~np.array(list(file_content[16:800])).reshape(28,28).astype(np.uint8)
#plt.imshow(image, cmap='gray')
#plt.show() - Used to display image in pop up window

# Reading labels from data files
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    labels = f.read()
int.from_bytes(labels[8:9], byteorder="big")
# print(labels) - Test labels are working

# ----------------------- Testing Neural Networks --------------------------------
# Single Neuron Network
# Create a new neural network.
m = kr.models.Sequential()
# Add a single neuron in a single layer, initialised with weight 2 and bias 1.
m.add(kr.layers.Dense(1, input_dim=1, activation="linear", kernel_initializer=kr.initializers.Constant(value=2), bias_initializer=kr.initializers.Constant(value=1)))
# Compile the model.
m.compile(loss="mean_squared_error", optimizer="sgd")
# Create some input values.
x = np.arange(0.0, 10.0, 1)
# Run each x value through the neural network.
y = m.predict(x)
# Plot the values.
plt.plot(x, y, 'k.')
#plt.show() - Used to display graph in pop up window

# Two Neuron Network
# Create a new neural network.
m = kr.models.Sequential()
# Add a two neurons in a single layer.
m.add(kr.layers.Dense(2, input_dim=1, activation="linear"))
# Add a single neuron in a single layer, initialised with weight 1 and bias 0.
m.add(kr.layers.Dense(1, activation="linear", kernel_initializer=kr.initializers.Constant(value=1), bias_initializer=kr.initializers.Constant(value=0)))
# Set the weight/bias of the two neurons.
m.layers[0].set_weights([np.matrix([2, 3]), np.array([-5, -3])])
# Compile the model.
m.compile(loss="mean_squared_error", optimizer="sgd")
# Create some input values.
x = np.arange(0.0, 10.0, 1)
# Run each x value through the neural network.
y = m.predict(x)
# Plot the values.
#plt.plot(x, y, 'k.')
#plt.show() - Used to display graph in pop up window

# Sigmoid Neuron Network
# Create a new neural network.
m = kr.models.Sequential()
# Add a single neuron in a single layer, initialised with weight 1 and bias 0, with sigmoid activation.
m.add(kr.layers.Dense(1, input_dim=1, activation="sigmoid", kernel_initializer=kr.initializers.Constant(value=1), bias_initializer=kr.initializers.Constant(value=0)))
# Compile the model.
m.compile(loss="mean_squared_error", optimizer="sgd")
# Create some input values.
x = np.arange(-10.0, 10.0, 1)
# Run each x value through the neural network.
y = m.predict(x)
# Plot the values.
plt.plot(x, y, 'k.')
#plt.show() - Used to display graph in pop up window
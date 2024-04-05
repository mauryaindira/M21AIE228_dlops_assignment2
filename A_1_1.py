import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Generate input data
x = np.linspace(-5, 5, 100)

# Compute outputs for each activation function
sigmoid_y = sigmoid(x)
relu_y = relu(x)
leaky_relu_y = leaky_relu(x)
tanh_y = tanh(x)

# Generate plots for each activation function
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, relu(x), label='ReLU')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('leaky_relu(x)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, tanh(x), label='Tanh')
plt.title('Tanh Activation Function')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input data
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# Generate sigmoid graph for the input data
x = np.array(random_values)
y = sigmoid(x)

# Plot
plt.plot(x, y, marker='o', linestyle='-')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()
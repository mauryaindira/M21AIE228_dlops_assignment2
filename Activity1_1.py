import numpy as np

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha*x, x)

def tanh(x):
    return np.tanh(x)

def apply_activation_functions(data):
    relu_output = relu(data)
    leaky_relu_output = leaky_relu(data)
    tanh_output = tanh(data)
    return relu_output, leaky_relu_output, tanh_output

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

relu_output, leaky_relu_output, tanh_output = apply_activation_functions(np.array(random_values))

print("ReLU output:", relu_output)
print("Leaky ReLU output:", leaky_relu_output)
print("Tanh output:", tanh_output)
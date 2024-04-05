import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
x = np.array(random_values)
y = sigmoid(x)

plt.plot(x, y, marker='o', linestyle='-', color='b')
plt.xlabel('Input')
plt.ylabel('Sigmoid Output')
plt.title('Sigmoid Function Graph')
plt.grid(True)
plt.show()

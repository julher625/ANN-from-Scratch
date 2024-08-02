import numpy as np

inputs = np.array([1.2, 5.1, 2.1])
weights = np.array([3.1, 2.1, 8.7])
bias = 3

outputs = np.dot(inputs, weights) + bias

print(outputs)
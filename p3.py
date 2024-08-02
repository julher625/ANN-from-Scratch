import numpy as np

inputs = np.array([1, 2, 3, 2.5])
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]) #Three neurons 

bias = [2,3,0.5]

outputs = np.dot(weights,inputs) + bias

print(outputs)
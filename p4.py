import numpy as np

inputs = np.array([[1, 2, 3, 2.5],
                    [2, 5, -1, 2],
                    [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]) #Three neurons 

bias = [2,3,0.5]

weights2 = np.array([[0.1, -0.14, 0.5],
                    [-0.5, 0.12, -0.33],
                    [-0.44, 0.73, -0.13]]) #Three neurons 

bias2 = [-1,2,-0.5]

outputs_l1 = np.dot(inputs, weights.T) + bias
outputs_l2 = np.dot(outputs_l1, weights2.T) + bias2

print(outputs_l1)
print(outputs_l2)
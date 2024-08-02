import numpy as np
np.random.seed(0)

class ActivactionFunction:
    def forward(self):
        pass

class ReLu(ActivactionFunction):
    def forward(self, input):
        self.output = np.maximum(0, input)
        return self.output

class Layer:
    def __init__(self):
        self.output = None
        self.activationFunction = None
        
class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons, activationFunction = ReLu()):
        weights = np.random.randn(n_inputs, n_neurons)
        min_val = np.min(weights)
        max_val = np.max(weights)
        self.weights = 2 * (weights - min_val) / (max_val - min_val) - 1
        self.biases = np.zeros((1, n_neurons))
        self.activationFunction = activationFunction
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.output = self.activationFunction.forward(self.output)

class Layer_Input(Layer):
    def __init__(self, X):
        self.output = X

class ANN:
    def __init__(self, input):
        self.layers: list[Layer] = [Layer_Input(input)]
    
    def forward(self):
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output)




X = np.array([[1, 2, 3, 2.5],
                    [2, 5, -1, 2],
                    [-1.5, 2.7, 3.3, -0.8]])

ann = ANN(X)
ann.layers.append(Layer_Dense(4,5))
ann.layers.append(Layer_Dense(5,2))

ann.forward()

print(ann.layers[1].output)
print(ann.layers[2].output)
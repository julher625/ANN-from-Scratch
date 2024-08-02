import numpy as np
# import math

# np.random.seed(0)

class ActivationFunction:
    def forward(self, input):
        pass

class ReLu(ActivationFunction):
    def forward(self, input):
        self.output = np.maximum(0, input)
        return self.output

class SoftMax(ActivationFunction):
    def forward(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

class Layer:
    def __init__(self):
        self.output = None
        self.activationFunction = None

class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons, activationFunction=ReLu()):
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

class Loss():
    def calculateLoss(self, predicted, target):
        self.loss = -(np.log(predicted)*target)
        return self.loss

class MeanSquare(Loss):
    def calculateLoss(self, predicted, target):
        self.loss = np.mean(np.square(target-predicted),axis=1)
        return self.loss

class ANN:
    
    isOneShot = True
    
    def __init__(self, X, y, batchSize = 10, loss = Loss()):
        self.layers = [Layer_Input(X[:batchSize])]
        self.loss = loss
        self.X = X
        self.y = y
        self.batchSize = batchSize

    def forward(self):
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i - 1].output)
        # max_indices = np.argmax(self.layers[-1].output, axis=1)
        # self.output = np.zeros_like(self.layers[-1].output)
        # self.output[np.arange(self.layers[-1].output.shape[0]), max_indices] = 1
        self.lossValue = self.loss.calculateLoss(self.layers[-1].output, self.y[:self.batchSize])
        

# X = np.array([[1, 2, 3, 2.5],
#               [2, 5, -1, 2],
#               [-1.5, 2.7, 3.3, -0.8]])

X = np.linspace(0, 2 * np.pi, 100).reshape(100, 1)
y = np.sin(X)
batchSize = 4
ann = ANN(X, y, batchSize=batchSize, loss=MeanSquare())
ann.layers.append(Layer_Dense(1, 10))
ann.layers.append(Layer_Dense(10, 5))
ann.layers.append(Layer_Dense(5, 1, SoftMax()))

ann.forward()

# print(ann.layers[1].output)
# print(ann.layers[2].output)
print(ann.lossValue)

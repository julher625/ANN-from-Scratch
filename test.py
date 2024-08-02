import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ActivationFunction:
    def forward(self, input):
        pass

    def backward(self, dvalues):
        pass

class ReLu(ActivationFunction):
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (self.input > 0)
        return self.dinputs
    
class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def forward(self, input):
        self.input = input
        self.output = np.where(input > 0, input, self.alpha * input)
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = dvalues * np.where(self.input > 0, 1, self.alpha)
        return self.dinputs

class Layer:
    def __init__(self):
        self.output = None
        self.activationFunction = None

class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons, activationFunction=ReLu()):
        self.weights = np.random.uniform(-1, 1, (n_inputs, n_neurons))
        self.biases = -np.zeros((1, n_neurons))
        self.activationFunction = activationFunction
        print(self.weights)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        self.output = self.activationFunction.forward(self.output)
    
    def backward(self, dvalues):
        self.dactivation = self.activationFunction.backward(dvalues)
        self.dweights = np.dot(self.inputs.T, self.dactivation)
        self.dbiases = np.sum(self.dactivation, axis=0, keepdims=True)
        self.dinputs = np.dot(self.dactivation, self.weights.T)
        return self.dinputs

class Layer_Input(Layer):
    def __init__(self, X):
        self.output = X

class Loss:
    def calculate(self, predicted, target):
        pass
    
    def backward(self, predicted, target):
        pass

class MeanSquare(Loss):
    def calculate(self, predicted, target):
        return np.mean(np.square(target - predicted))
    
    def backward(self, predicted, target):
        return -2 * (target - predicted) / target.shape[0]

class ANN:
    def __init__(self, X, y, batchSize=10, loss=MeanSquare()):
        self.layers = [Layer_Input(X[:batchSize])]
        self.loss = loss
        self.X = X
        self.y = y
        self.batchSize = batchSize
        self.loss_history = []
        self.prediction_history = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self): 
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i - 1].output)

    def backward(self):
        dvalues = self.loss.backward(self.layers[-1].output, self.y_batch)
        for i in reversed(range(1, len(self.layers))):
            dvalues = self.layers[i].backward(dvalues)

    def train(self, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            epoch_predictions = []
            for start in range(0, len(self.X), self.batchSize):
                end = start + self.batchSize
                self.layers[0] = Layer_Input(self.X[start:end])
                self.y_batch = self.y[start:end]
                self.forward()
                self.backward()
                for layer in self.layers[1:]:
                    if isinstance(layer, Layer_Dense):
                        layer.weights -= learning_rate * layer.dweights
                        layer.biases -= learning_rate * layer.dbiases
                epoch_predictions.extend(self.layers[-1].output)
            
            loss = self.loss.calculate(self.layers[-1].output, self.y_batch)
            self.loss_history.append(loss)
            if epoch % 10000 == 0:
                self.prediction_history.append(np.array(epoch_predictions).flatten())
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        self.layers[0] = Layer_Input(X)
        self.forward()
        return self.layers[-1].output

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.show()

# Inicializando la red y entrenando
X = np.linspace(0, 2 * np.pi, 100).reshape(100, 1)
y = np.sin(X)+2*np.cos(10*X)
batchSize = 100
ann = ANN(X, y, batchSize=batchSize, loss=MeanSquare())
ann.add_layer(Layer_Dense(1, 16, activationFunction=LeakyReLU()))
ann.add_layer(Layer_Dense(16, 32, activationFunction=LeakyReLU()))
ann.add_layer(Layer_Dense(32, 64, activationFunction=LeakyReLU()))
ann.add_layer(Layer_Dense(64, 64, activationFunction=LeakyReLU()))
ann.add_layer(Layer_Dense(64, 1, activationFunction=LeakyReLU()))

yy=ann.predict(X)
print("Outputs")
for layer in ann.layers:
    print(layer.output)

print("Predicted")
print(yy)
print("red")
print(ann.layers[-1].output)
ann.train(epochs=1000000, learning_rate=0.0001)


# Crear la animación
fig, ax = plt.subplots()
line_real, = ax.plot(X, y, label='Real', linestyle='--', color='blue')
line_pred, = ax.plot(X, ann.prediction_history[0], label='Predicted', linestyle='-', color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Predicciones del modelo durante el entrenamiento')
ax.legend()

def update(frame):
    line_pred.set_ydata(ann.prediction_history[frame])
    return line_pred,

ani = animation.FuncAnimation(fig, update, frames=len(ann.prediction_history), blit=True)
ani.save("ann_training_animation_2.mp4", writer='ffmpeg')
# plt.show()

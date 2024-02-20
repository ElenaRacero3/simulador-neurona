import numpy as np

# OBJETO

class Neuron:
    def __init__(self, weights, bias, func):
        self.weights = np.array(weights)
        self.bias = bias
        self.func = func

    def changeWeights(self, new_weights):
        self.weights = np.array(new_weights)

    def changeBias(self, new_bias):
        self.bias = new_bias

    def run(self, input_data):
        input_data = np.array(input_data)
        output = np.dot(self.weights, input_data) + self.bias
        return getattr(Neuron, f"_Neuron__{self.func}")(output)

    @staticmethod
    def __relu(x):
        return max(0, x)

    @staticmethod
    def __sigmoide(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __tangente(x):
        return np.tanh(x)


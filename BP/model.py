import numpy as np

np.random.seed(10)


class Model:
    def __init__(self):
        self.layer1 = Layer1(2, 24, 'tanh', 0.001)
        self.layer2 = Layer2(25, 1, 'phi', 0.001)
        self.prediction = 0

    def forward(self, data):
        # layer1
        out_1 = self.layer1.forward(data)
        in_2 = np.append(out_1, np.ones(1))
        # layer2
        out_2 = self.layer2.forward(in_2)
        return out_2

    def backward(self, delta):
        delta = self.layer2.backward(delta)
        self.layer1.backward(delta)


class Layer:
    def __init__(self, size_in, size_out, activation, lr):
        self.weight = np.random.uniform(1, -1, (size_in, size_out))
        self.activation = activation
        self.lr = lr

    def forward(self, data):
        # xi is the input of each layer
        self.xi = data
        self.z = np.matmul(self.xi, self.weight)
        if self.activation == 'tanh':
            self.a = tanh(self.z)
        else:
            self.a = self.z
        return self.a


class Layer1(Layer):
    def backward(self, delta):
        delta = delta.T
        data_in = self.xi.reshape(2, 1)
        weight_update = np.matmul(data_in, delta)
        derivatives = tanh_prime(self.z)
        # update
        weight_update = derivatives * weight_update
        self.weight = self.weight + self.lr * weight_update


class Layer2(Layer):
    def backward(self, delta):
        delta = delta.reshape(1, 1)
        data_in = self.xi.reshape(25, 1)
        weight_update = np.matmul(data_in, delta)
        # update
        self.weight = self.weight + self.lr * weight_update
        delta = np.matmul(self.weight, delta)
        delta = delta[:-1, :]
        return delta




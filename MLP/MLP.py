import numpy as np


def sigmoid(data_in):
    data_out = 1 / (1 + np.exp(-data_in))
    return data_out


class Model:
    def __init__(self, input_size, activation):
        self.input_num = input_size
        self.activation = activation
        self.w = np.random.normal(scale=0.1, size=(input_size, 10))

    def forward(self, data_in):
        data_out = np.matmul(data_in, self.w)
        if self.activation == 'sigmoid':
            data_activated = sigmoid(data_out)
        else:
            raise Exception('Wrong type of activation function!')
        self.data_activated = data_activated
        return data_activated

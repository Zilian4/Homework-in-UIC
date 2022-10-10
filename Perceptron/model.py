import numpy as np


class Perceptron:
    def __init__(self):
        np.random.seed(4)
        self.w0 = np.random.uniform(-1 / 4, 1 / 4)
        self.w1 = np.random.uniform(-1, 1)
        self.w2 = np.random.uniform(-1, 1)
        self.w = np.array([self.w1, self.w2, self.w0])
        self.loss = 1
        self.loss_list = []  # record all the losses to plot the line chart in the end

    def forward(self, data_in):  # calculate the output(before activation function) and the prediction(after)
        data_out = np.matmul(self.w, data_in)
        prediction = self.step_function(data_out)
        return data_out, prediction

    def update_loss(self, errors):  # current errors is the loss of current epoch
        self.loss = errors
        self.loss_list.append(errors)

    def step_function(self, num_in):
        if num_in >= 0:
            num_out = 1
        else:
            num_out = 0
        return num_out

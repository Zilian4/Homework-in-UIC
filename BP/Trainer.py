import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, data, target):

        self.original_data = data.reshape((300, 1))
        self.data = np.column_stack((self.original_data, np.ones((300, 1))))
        self.target = target
        self.model = model
        self.losses = []
        self.epsilon = 0.01

    def start_train(self):
        i = 0
        loss = 1
        while loss>self.epsilon:
            print("epoch:", i)
            i+=1
            self.predict_list = []
            # count loss
            for i in range(len(self.target)):
                x = self.data[i]
                predict = self.model.forward(x)
                self.predict_list.append(predict)
            prediction = np.reshape(self.predict_list, 300)
            # calculate the total loss
            loss = MSE(prediction, self.target)
            self.losses.append(loss)
            print(loss)
            # update
            for i in range(len(self.data)):
                x = self.data[i]
                y = self.target[i]
                predict = self.model.forward(x)
                delta = y - predict
                self.model.backward(delta)

    def plot_prediction(self):
        plt.scatter(self.original_data, self.predict_list, s=20)
        plt.title('xi_di')
        plt.show()



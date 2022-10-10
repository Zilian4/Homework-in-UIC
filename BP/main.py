import numpy as np
import matplotlib.pyplot as plt


def MSE(prediction, target):
    prediction = np.array(prediction).reshape(300, )
    mse_loss = np.mean((target - prediction) ** 2)
    return mse_loss


def tanh(x):
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return y


def tanh_prime(x):
    y = 1 - (tanh(x) ** 2)
    return y


class Dataloader:
    def __init__(self):
        self.X = np.random.uniform(0, 1, 300)
        self.V = np.random.uniform(-0.1, 0.1, 300)
        self.di = np.sin(20 * self.X) + 3 * self.X + self.V

    def plot_data(self):
        plt.scatter(self.X, self.di, s=20)
        plt.title('original data')
        plt.show()

    def get_data(self):
        return self.X, self.di


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = Dataloader()
    data.plot_data()

    w1 = np.random.uniform(1, -1, (2, 24))
    lr1 = 0.01
    w2 = np.random.uniform(1, -1, (24, 1))
    lr2 = 0.01
    b = np.random.uniform(-1, 1)

    epsilon = 0.02

    data_input, target = data.get_data()
    data_input = np.column_stack((data_input.reshape(300, 1), np.ones((300, 1))))

    keep_training = True
    epoch = 0
    losses = []
    while keep_training:
        #  count loss
        predictions = []
        for i in range(len(target)):
            x = data_input[i].reshape(1, 2)
            y = target[i]

            # forward
            out_1 = np.matmul(x, w1)
            out_1_a = tanh(out_1)  # 24
            prediction = np.matmul(out_1_a, w2) + b  # 1x1
            predictions.append(prediction)
        loss = MSE(predictions, target)
        losses.append(loss)
        # show the current epoch and loss
        if epoch % 100 == 0:
            print("epoch:", epoch, 'loss:', loss)
        epoch += 1
        if loss <= epsilon:
            keep_training = False

        for i in range(len(data_input)):
            x = data_input[i].reshape(1, 2)
            y = target[i]

            # forward
            out_1 = np.matmul(x, w1)
            out_1_a = tanh(out_1)  # 24
            out_2 = np.matmul(out_1_a, w2) + b  # 1x1

            # backward
            delta_2 = y - out_2  # 1x1
            delta_1 = np.matmul(w2, delta_2)  # 24x1

            # update
            w2_update = (out_1_a.T * delta_2)
            w2 = w2 + lr2 * w2_update
            b = b + lr2 * delta_2
            w1_update = np.matmul(tanh_prime(out_1).T * delta_1, x).T
            w1 = w1 + lr1 * w1_update

    target = plt.scatter(data.get_data()[0], data.get_data()[1], s=20)
    predict = plt.scatter(data.get_data()[0], predictions, s=20, c='r')
    plt.legend([target, predict], ["target", "predict"], loc='upper right', scatterpoints=1)
    plt.title('desire output and prediction')
    plt.show()
    plt.plot(range(len(losses)), losses)
    plt.title('Loss_epoch')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, learning_rate, n, epsilon):
        self.model = model
        self.epochs = 0
        self.errors = []
        self.learning_rate = learning_rate
        self.accuracy = 0
        self.test_accuracy = 0
        self.n = n
        self.epsilon = epsilon

    def start_train(self, train_data, train_label):
        delta_acc = 0.5
        train_data, train_label = train_data[:self.n], train_label[:self.n]
        while self.accuracy < 0.9 and delta_acc > self.epsilon:
            self.epochs += 1
            errors = 0
            # count errors
            for j in range(len(train_data)):
                data = train_data[j].reshape(1, 784)
                label = train_label[j]
                predict = self.model.forward(data)
                if not np.argmax(predict) == np.argmax(label):
                    errors += 1
            self.errors.append(errors)
            accuracy = 1 - (errors / len(train_data))
            delta_acc = abs(accuracy - self.accuracy)
            self.accuracy = accuracy

            # update weight
            for j in range(len(train_data)):
                data = train_data[j].reshape(1, 784)
                label = train_label[j]
                predict = self.model.forward(data)
                self.model.w = self.model.w + self.learning_rate * np.matmul(data.transpose(), (label - predict))
            print('epoch:', self.epochs, 'accuracy:', self.accuracy)
        print("accuracy in training:", self.accuracy)

    def start_test(self, test_data, test_label):
        errors = 0
        for i in range(len(test_data)):
            data = test_data[i].reshape(1, 784)
            label = test_label[i]
            predict = self.model.forward(data)
            if not np.argmax(predict) == np.argmax(label):
                errors += 1
        accuracy = 1 - (errors / 10000)
        self.test_accuracy = accuracy
        print('accuracy in testing data:', accuracy)

    def plot_errors_epoch(self):
        plt.rc('axes', facecolor='white')
        plt.rc('figure', figsize=(6, 4))
        plt.rc('axes', grid=False)
        plt.plot(range(self.epochs), self.errors, '.:r')

        plt.title(
            'Errors-Epochs( learning rate={0},n = {1} ,epsilon:{2})'.format(self.learning_rate, self.n, self.epsilon))
        plt.xlabel('Epochs')
        plt.ylabel('Errors')
        plt.show()

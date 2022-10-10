import numpy as np
import matplotlib.pyplot as plt
import gzip

# np.random.seed(10)
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


def step(data_in):
    data_in = data_in+np.abs(data_in)
    data_in = data_in/data_in
    data_out = np.nan_to_num(data_in)
    return data_out


class Model:
    def __init__(self, input_size, activation):
        self.input_num = input_size
        self.activation = activation
        self.w = np.random.normal(scale=0.1, size=(input_size, 10))

    def forward(self, data_in):
        data_out = np.matmul(data_in, self.w)
        if self.activation == 'step':
            data_activated = step(data_out)
        else:
            raise Exception('Wrong type of activation function!')
        self.data_activated = data_activated
        return data_activated


class Dataloader:

    def get_train_img(self):
        train_image = gzip.open(r'data/train-images-idx3-ubyte.gz', 'r')
        train_image = train_image.read()[16:]
        train_image = list(train_image)
        train_image = np.reshape(train_image, (60000, 784))
        return train_image

    def get_test_img(self):
        test_image = gzip.open(r'data/t10k-images-idx3-ubyte.gz', 'r')
        test_image = test_image.read()[16:]
        test_image = list(test_image)
        test_image = np.reshape(test_image, (10000, 28, 28))
        return test_image

    def get_train_label(self):
        train_label = gzip.open(r'data/train-labels-idx1-ubyte.gz', 'r')
        train_label = train_label.read()[8:]
        train_label = list(train_label)
        train_label = np.eye(10)[train_label]
        return train_label

    def get_test_label(self):
        test_label = gzip.open(r'data/t10k-labels-idx1-ubyte.gz', 'r')
        test_label = test_label.read()[8:]
        test_label = list(test_label)
        test_label = np.eye(10)[test_label]
        return test_label

    def show_sample(self):
        sample_image = gzip.open(r'data/t10k-images-idx3-ubyte.gz', 'r')
        sample_image = sample_image.read()[16:16 + 784]
        sample_image = list(sample_image)
        sample_image = np.reshape(sample_image, (28, 28))
        plt.imshow(sample_image)
        plt.show()


dataloader = Dataloader()
train_data = dataloader.get_train_img()
train_label = dataloader.get_train_label()
test_data = dataloader.get_test_img()
test_label = dataloader.get_test_label()

lr = 0.0000005
n = 600000
epsilon = 0.0001
MLP = Model(input_size=784, activation='step')
T = Trainer(MLP, lr, n, epsilon)
T.start_train(train_data, train_label)
T.start_test(test_data, test_label)
T.plot_errors_epoch()

import numpy as np
import matplotlib.pyplot as plt


class Dataset:

    def __init__(self, dataset_scale):
        np.random.seed(5)
        self.w0 = np.random.uniform(-1, 1)
        self.w1 = np.random.uniform(-1, 1)
        self.w2 = np.random.uniform(-1, 1)
        self.s_0 = []
        self.s_1 = []
        self.data = np.random.uniform(-1, 1, [dataset_scale, 2])
        self.data = np.insert(self.data, 2, np.ones(dataset_scale), axis=1)

        target = []

        for i in range(len(self.data)):
            current_set = self.data[i]
            if self.w0 + self.w1 * current_set[0] + self.w2 * current_set[1] >= 0:
                r = self.w0 + self.w1 * current_set[0] + self.w2 * current_set[1]
                self.s_1.append(current_set)
                target.append(1)
            else:
                self.s_0.append(current_set)
                target.append(0)

        if len(self.s_0) == 0 or len(self.s_1) == 0:
            raise Exception('There is only one class in the data,please reset the random seeds')

        self.s_1 = np.array(self.s_1)
        self.s_0 = np.array(self.s_0)
        # target is class of data
        self.target = np.array(target)

    '''Show the distribution of the data and the boundary of S1 & S0'''

    def get_distribution(self):
        s_0 = self.s_0
        s_1 = self.s_1
        s0 = plt.scatter(s_0[:, 0], s_0[:, 1], s=20, c='red', marker='v')
        s1 = plt.scatter(s_1[:, 0], s_1[:, 1], s=20, c='blue', marker='^')

        x = np.linspace(-1, 1, 10)
        y = (self.w0 + self.w1 * x) / -self.w2
        ideal_model = plt.plot(x, y)
        ideal_model = plt.legend(ideal_model, ['ideal_model'], loc='lower left')

        plt.xlim(-1.25, 1.35)
        plt.ylim(-1.35, 1.25)
        plt.xticks()
        plt.yticks()
        plt.tick_params(axis='x', colors='black')
        plt.tick_params(axis='y', colors='red')

        plt.title("Distribution of data")
        plt.legend([s1, s0], ["S1", "S0"], loc='upper right', scatterpoints=1)
        plt.gca().add_artist(ideal_model)
        plt.show()

    def get_data(self):
        return self.data, self.target


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


class Trainer:
    def __init__(self, data, P: {Perceptron}, learning_rate):
        self.data = data[0]
        self.targets = data[1]
        self.perceptron = P
        self.lr = learning_rate
        self.epoch = 1

    def start_train(self):
        print('Start training........')
        while self.perceptron.loss != 0:
            errors = 0
            print("Current weight -----> w0:{0}: ,w1:{1} ,w2:{2}\n".format(self.perceptron.w[2], self.perceptron.w[0],
                                                                           self.perceptron.w[1]))
            for i in range(len(self.data)):
                current_point = self.data[i]
                target = self.targets[i]
                data_out, prediction = self.perceptron.forward(data_in=current_point)
                if prediction > target:
                    self.perceptron.w = self.perceptron.w - self.lr * current_point
                    errors += 1
                elif prediction < target:
                    self.perceptron.w = self.perceptron.w + self.lr * current_point
                    errors += 1
            self.perceptron.update_loss(errors)
            print("-=========epoch{0}===========-\nCurrent loss:{1} ".format(self.epoch, self.perceptron.loss))
            self.epoch += 1
        print('-=========Finished !!=========-')

    def plot_loss_img(self):
        plt.rc('axes', facecolor='white')
        plt.rc('figure', figsize=(6, 4))
        plt.rc('axes', grid=False)
        plt.plot(range(1, self.epoch), self.perceptron.loss_list, '.:r')

        plt.title('Loss-Epoch( learning rate={0} )'.format(self.lr))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def make_comparison(self, ideal_weight):
        w0_ = self.perceptron.w[2]
        w1_ = self.perceptron.w[0]
        w2_ = self.perceptron.w[1]
        print("Current weight(after training) ===> w0:{0}: ,w1:{1} ,w2:{2}\n".format(w0_, w1_, w2_))
        print('ratio of expect weights and real weights:w0:{0}: ,w1:{1} ,w2:{2}'.format(w0_ / -0.2, w1_ / 0.6, w2_ / 1))

        x = np.linspace(-1, 1, 10)
        y = (ideal_weight[0] + ideal_weight[1] * x) / -ideal_weight[2]
        ideal_model = plt.plot(x, y)
        ideal_model = plt.legend(ideal_model, ['ideal_model'], loc='lower left')

        x_ = np.linspace(-1, 1, 10)
        y_ = (w1_ * x + w0_) / -w2_
        trained_model = plt.plot(x_, y_)
        plt.legend(trained_model, ['trained_model'], loc='upper right')
        plt.gca().add_artist(ideal_model)

        plt.xlim(-1.25, 1.35)
        plt.ylim(-1.35, 1.25)
        plt.xticks()
        plt.yticks()
        plt.tick_params(axis='x', colors='black')
        plt.tick_params(axis='y', colors='red')
        plt.show()


'''Setting hyper-parameters before training'''

lr = 0.01  # learning rate
n = 1000  # scale of n
'''Instantiate Dataset, Trainer and Perceptron'''
D = Dataset(dataset_scale=n)
D.get_distribution()  # show the distribution of the dataset

P = Perceptron()
T = Trainer(D.get_data(), P, learning_rate=lr)
T.start_train()  # start training
T.plot_loss_img()  # plot the image of loss
T.make_comparison(ideal_weight=[D.w0, D.w1, D.w2])  # make comparison about trained model and ideal model

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
        print(self.target)

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

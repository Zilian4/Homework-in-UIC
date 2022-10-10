import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)


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

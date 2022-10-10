import gzip
import numpy as np
import matplotlib.pyplot as plt


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

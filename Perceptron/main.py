from dataset import *
from train import *
from model import *


if __name__ == '__main__':
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

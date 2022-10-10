from MLP import *
from Trainer import *
from Data import *

if __name__ == '__main__':
    dataloader = Dataloader()
    train_data = dataloader.get_train_img()
    train_label = dataloader.get_train_label()
    test_data = dataloader.get_test_img()
    test_label = dataloader.get_test_label()

    lr = 0.000001
    n = 600000
    epsilon = 0.00005
    MLP = Model(input_size=784, activation='sigmoid')
    T = Trainer(MLP, lr, n, epsilon)
    T.start_train(train_data, train_label)
    T.start_test(test_data, test_label)
    T.plot_errors_epoch()

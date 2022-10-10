from model import *
import matplotlib.pyplot as plt


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

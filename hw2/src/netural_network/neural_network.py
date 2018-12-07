import numpy as np
import random
import matplotlib.pyplot as plt

from .full_connected_layer import *
from .sigmoid_activator import *
from .softmax_activator import *


class NeuralNetwork(object):

    def __init__(self, layer_nodes):
        self.__layers = []
        self.__deep = len(layer_nodes) - 1
        for i in range(self.__deep - 1):
            self.__layers.append(FullConnectedLayer((layer_nodes[i], layer_nodes[i + 1]), SigmoidActivator()))
        self.__layers.append(FullConnectedLayer((layer_nodes[self.__deep - 1], layer_nodes[self.__deep]), SoftmaxActivator()))

    def init(self):
        self.train_loss = []
        self.evalu_loss = []

    def predict(self, x):
        y = x.T
        for layer in self.__layers:
            layer.forward(y)
            y = layer.output_array
        return y

    def __test(self, test_data_X, test_data_Y):
        n = len(test_data_X)
        acc = 0
        Y = self.predict(test_data_X)
        loss = self.calc_cross_entropy_loss(test_data_Y, Y)
        Y = Y.T
        for y, y_ in zip(Y, test_data_Y):
            if y.tolist().index(max(y)) == y_.tolist().index(max(y_)):
                acc += 1
        return acc, n, loss

    def train(self, X, Y, epochs, batch_size, learning_rate=0.01, test_rate=0.05):
        self.init()
        n = len(X)
        division = int((1 - test_rate) * n)
        pre_loss = 100
        stop_flag = 100
        train_data_X, train_data_Y = X[:division], Y[:division]
        test_data_X, test_data_Y = X[division:], Y[division:]
        for i in range(epochs):
            train_loss = 0
            rand = np.arange(0, division)
            np.random.shuffle(rand)
            train_data_X, train_data_Y = train_data_X[rand], train_data_Y[rand]
            batchs_X = [train_data_X[k:k+batch_size] for k in range(0, division - batch_size + 1, batch_size)]
            batchs_Y = [train_data_Y[k:k+batch_size] for k in range(0, division - batch_size + 1, batch_size)]
            for j in range(len(batchs_X)):
                train_loss += self.__train_batch(batchs_X[j], batchs_Y[j], learning_rate)
            train_loss /= division // batch_size
            if train_loss > pre_loss:
                stop_flag -= 1
                if stop_flag < 0:
                    return
            pre_loss = train_loss
            self.train_loss.append(train_loss)
            test_acc, test_n, eval_loss = self.__test(test_data_X, test_data_Y)
            self.evalu_loss.append(eval_loss)
            print('Epoch %d : %d / %d => %f with loss: %f' % (i, test_acc, test_n, test_acc / test_n, train_loss))

    def __train_batch(self, batch_X, batch_Y, learning_rate):
        output_array = self.predict(batch_X)
        loss = self.calc_cross_entropy_loss(batch_Y, output_array)
        self.calc_gradient(batch_Y)
        self.update_weight(learning_rate)
        return loss

    def calc_cross_entropy_loss(self, batch_Y, output_array):
        output_array = output_array.T
        loss = batch_Y * np.log(output_array)
        loss = np.sum(loss) / batch_Y.shape[0]
        return float(-loss)

    def calc_gradient(self, label):
        label = label.T
        delta = self.__layers[-1].activator.backward(self.__layers[-1].output_array) * (label - self.__layers[-1].output_array)
        for layer in self.__layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.__layers:
            layer.update(rate)

    def show_diagram(self):
        plt.title('loss diagram')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(range(len(self.train_loss)), self.train_loss, label='train_loss')
        plt.plot(range(len(self.train_loss)), self.evalu_loss, label='evalu_loss')
        plt.legend()
        plt.show()

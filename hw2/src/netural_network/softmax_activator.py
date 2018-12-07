import numpy as np


class SoftmaxActivator(object):

    @staticmethod
    def forward(k):
        exp = np.exp(k)
        exp_sum = np.sum(exp, axis=0, keepdims=True)
        return exp / exp_sum

    @staticmethod
    def backward(k):
        return np.ones(k.shape)

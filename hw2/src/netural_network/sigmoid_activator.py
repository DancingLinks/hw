import numpy as np


class SigmoidActivator(object):

    @staticmethod
    def forward(k):
        k = -k
        k = k * (k <= 700) + 700 * (k > 700)
        return 1.0 / (1.0 + np.exp(k))

    @staticmethod
    def backward(k):
        return k * (1 - k)

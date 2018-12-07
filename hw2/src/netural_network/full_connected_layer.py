import numpy as np


class FullConnectedLayer(object):

    def __init__(self, shape, activator):
        self.in_size = shape[0]
        self.out_size = shape[1]
        self.activator = activator
        self.w = np.random.uniform(-0.1, 0.1, (shape[1], shape[0]))
        # self.w = np.random.randn(shape[1], shape[0])
        self.b = np.random.uniform(-0.1, 0.1, (shape[1], 1))
        # self.b = np.zeros((shape[1], 1))
        # self.b = np.random.randn(shape[1], 1)

    def forward(self, input_array):
        self.input_array = input_array  # input : batch
        self.output_array = self.activator.forward(np.matmul(self.w, input_array) + self.b) # output : batch

    def backward(self, delta):
        self.delta = self.activator.backward(self.input_array) * np.matmul(self.w.T, delta) # input : batch
        self.delta_w = np.matmul(delta, self.input_array.T) # output : input
        self.delta_b = np.sum(delta, axis=1, keepdims=True) # output : 1

    def update(self, learnging_rate):
        self.w += learnging_rate * self.delta_w
        self.b += learnging_rate * self.delta_b

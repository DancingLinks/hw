from netural_network.neural_network import NeuralNetwork
import numpy as np


X = []
Y = []

if __name__ == '__main__':

    input_file = open('wine.data', 'r+')
    
    for line in input_file.readlines():
        split = line.split(',')
        X.append([float(x) for x in split[1:]])
        Y.append([1 if int(split[0]) == label else 0 for label in range(1,4)])
    input_file.close()

    X = np.array(X)
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    Y = np.array(Y)
    rand = np.arange(0, len(X))
    np.random.shuffle(rand)
    X = X[rand]
    Y = Y[rand]

    nn = NeuralNetwork([13, 32, 3])
    nn.train(X, Y, epochs=10000, batch_size=64, learning_rate=0.01, test_rate=0.2)
    nn.show_diagram()

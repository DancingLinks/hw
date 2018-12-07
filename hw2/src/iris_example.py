from netural_network.neural_network import NeuralNetwork
import numpy as np


LABELS = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
X = []
Y = []


if __name__ == '__main__':

    input_file = open('iris.data', 'r+')
    
    for line in input_file.readlines():
        split = line.split(',')
        X.append([float(x) for x in split[:-1]])
        Y.append([1 if split[-1][:-1] == label else 0 for label in LABELS])
    
    input_file.close()
    
    X, Y = np.array(X), np.array(Y)
    rand = np.arange(0, len(X))
    np.random.shuffle(rand)
    X, Y = X[rand], Y[rand]

    nn = NeuralNetwork([4, 32, 3])
    nn.train(X, Y, epochs=100, batch_size=8, learning_rate=0.01, test_rate=0.2)
    nn.show_diagram()

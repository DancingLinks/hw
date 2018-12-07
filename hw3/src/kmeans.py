import numpy as np
from collections import defaultdict
from random import uniform
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Kmeans:

    __color = ['r', 'g', 'b', 'c', 'k', 'm', 'y']
    __dis_history = []

    def __init__(self, k):
    	self.__k = k


    # Load data from a csv type file
    def loadtxt(self, path, delimiter=',', usecols=None, dtype=float):
        self.__data = np.loadtxt(path, delimiter=delimiter, usecols=usecols, dtype=dtype)
        self.__dimensions = self.__data.shape[1]


    # Calculate the distance between a and b
    def __distance(self, a, b):
        _sum = 0.0
        for _a, _b in zip(a, b):
            _sum += (_a - _b) ** 2
        return _sum


    # Calculate the average distrance between points and center
    def __point_avg(self, points):
        points = np.array(points)
        return [sum(points[:, dimension]) / len(points) for dimension in range(self.__dimensions)]


    # Assign each point to one center
    def __assign_points(self, centers):
        assignments = []
        for point in self.__data:
            shortest, shortest_index = 1000000000, 0
            for i in range(len(centers)):
                val = self.__distance(point, centers[i])
                if val < shortest:
                    shortest, shortest_index = val, i
            assignments.append(shortest_index)
        self.__assignments = assignments


    # Calculate the new center and store the history of distance
    def __calculate_center_and_distance(self, means):
        _centers = [self.__point_avg(points) for _, points in means.items()]
        distance = 0.0
        for assignment, points in means.items():
            if assignment >= len(_centers):
                break
            for point in points:
                distance += self.__distance(_centers[assignment], point)
        self.__dis_history.append(distance)
        return _centers


    # Update centers
    def __update_centers(self):
        new_means = defaultdict(list)
        for assignment, point in zip(self.__assignments, self.__data):
            new_means[assignment].append(point)
        return self.__calculate_center_and_distance(new_means)


    def __generate(self):
        centers = []
        min_max = defaultdict(int)
        for x in self.__data:
            for i in range(self.__dimensions):
                val = x[i]
                min_key, max_key = 'min_%d' % i, 'max_%d' % i
                if min_key not in min_max or val < min_max[min_key]:
                    min_max[min_key] = val
                if max_key not in min_max or val > min_max[max_key]:
                    min_max[max_key] = val
        for _k in range(self.__k):
            centers.append([
                uniform(min_max['min_%d' % i], min_max['max_%d' % i]) for i in range(self.__dimensions)
            ])
        return centers


    def run(self):
        centers = self.__generate()
        self.__assign_points(centers)
        pre_assignments = None
        while pre_assignments != self.__assignments:
            new_centers = self.__update_centers()
            pre_assignments = self.__assignments
            self.__assign_points(new_centers)
        self.__centers = new_centers


    def show(self, n_components=2):
        pca = PCA(n_components=n_components)
        data = np.append(self.__centers, self.__data, axis=0)
        data = pca.fit_transform(data)
        centers, data = data[:self.__k], data[self.__k:]
        kmeans_x = [[] for _ in range(self.__k)]
        kmeans_y = [[] for _ in range(self.__k)]
        for assignment, point in zip(self.__assignments, data):
            kmeans_x[assignment].append(point[0])
            kmeans_y[assignment].append(point[1])
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.set_title('Scatter')
        for _k in range(self.__k):
            ax1.scatter(kmeans_x[_k], kmeans_y[_k], s=8, c=self.__color[_k % len(self.__color)])
            ax1.scatter(centers[_k][0], centers[_k][1], s=64, c=self.__color[_k % len(self.__color)], marker='x')
        ax2.set_title('Distance')
        ax2.plot(range(len(self.__dis_history)), self.__dis_history)
        plt.show()

from numpy import *
import matplotlib.pyplot as plt


class LogisticRegression:

	# Value for label_1 or label_0
	__LABEL_1 = ''
	__LABEL_0 = ''

	# Label of iris input data ( LABEL_1 or LABEL_0 )
	__label = []
	
	# dataset
	__data = []
	__rows = 0
	__cols = 0

	# logistic regression variables
	__beta = mat(zeros((3, 1)))
	__h = 0
	__theta_1 = 0
	__theta_2 = 0

	# variables for figure
	__figure_x_label = [[[],[]],[[],[]]]
	__x_range = [10000, 0]
	__y_range = []


	def __init__(self, label_0, label_1):
		self.__LABEL_0 = label_0
		self.__LABEL_1 = label_1


	def __append_data(self, data_line):
		split = data_line.split(',')
		this_label = split[-1:][0][:-1]
		split = list(map(float, split[:-1]))
		if this_label == self.__LABEL_0:
			self.__label.append([0])
			self.__figure_x_label[0][0].append(split[0])
			self.__figure_x_label[0][1].append(split[1])
		elif this_label == self.__LABEL_1:
			self.__label.append([1])
			self.__figure_x_label[1][0].append(split[0])
			self.__figure_x_label[1][1].append(split[1])
		self.__x_range[0] = min(self.__x_range[0], split[0])
		self.__x_range[1] = max(self.__x_range[1], split[0])
		self.__data.append(split[:2])


	def set_data_from_file(self, path):
		input_file = open(path, 'r+')
		for line in input_file.readlines():
			self.__append_data(line)
		input_file.close()
		self.__data = array(self.__data, dtype='float64')
		self.__data = mat(insert(self.__data, self.__data.shape[1], values=1, axis=1))
		self.__label = array(self.__label)
		self.__rows, self.__cols = self.__data.shape[0], self.__data.shape[1]


	def calculate(self):
		while True:
			self.__p = 1.0 / (1 + exp(-(self.__data * self.__beta)))
			self.__theta_1 = multiply(1.0 / self.__rows, self.__data.T * (self.__p - self.__label)) # (3, 1)
			self.__theta_2 = multiply(1.0 / self.__rows, self.__data.T * mat(diag(multiply(self.__p, (1-self.__p)).T.getA()[0])) * self.__data) # (3, 3)
			self.__beta = self.__beta - self.__theta_2.I * self.__theta_1
			if linalg.norm(self.__theta_1) < 0.000001:
				break
		self.__beta = self.__beta.T.getA()[0]


	def show(self, c0='r', c1='b'):
		plt.figure(1)
		self.__y_range.append(-(self.__beta[2] + self.__beta[0] * self.__x_range[0]) / self.__beta[1])
		self.__y_range.append(-(self.__beta[2] + self.__beta[0] * self.__x_range[1]) / self.__beta[1]) 
		plt.plot(self.__x_range, self.__y_range)
		plt.scatter(self.__figure_x_label[0][0], self.__figure_x_label[0][1], c=c0)
		plt.scatter(self.__figure_x_label[1][0], self.__figure_x_label[1][1], c=c1)
		plt.show()

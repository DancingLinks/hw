# 一个完整的example
# 用实际的类别名替换 LABEL_NAME_0 和 LABEL_NAME_1，例如'Iris-setosa', 'Iris-versicolor'
# 用数据集的本地路径替换 DATASET_PATH ，例如 'iris.data'

import logistic_regression as lr

if __name__ == '__main__':
	logisic_regression = lr.LogisticRegression('Iris-setosa', 'Iris-versicolor')
	logisic_regression.set_data_from_file('iris.data')
	logisic_regression.calculate()
	logisic_regression.show()

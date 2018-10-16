# A simple example

import logistic_regression as lr

if __name__ == '__main__':

	# Change LABEL_NAME_0 ans LABEL_NAME_1 into the value of label of your dataset, like'Iris-setosa' or 'Iris-versicolor'.
	logistic_regression = lr.LogisticRegression('Iris-setosa', 'Iris-versicolor')
	
	# Change DATASET_PATH into you path of dataset 'iris.data'.
	# After this step, the data will be loaded and initialized.
	logistic_regression.set_data_from_file('iris.data')
	
	# The calculate method is the implement of logistic regression with newton method.
	logistic_regression.calculate()

	# By this step, it can generate a simple diagram of the labels and vectors of input data and it can draw a line which represent the result of logistic regression.
	logistic_regression.show()

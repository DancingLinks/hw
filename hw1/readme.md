# Assignment 1 
auth: Qian yu 1831582



## Example：

```python
# 一个完整的example
# 用实际的类别名替换 LABEL_NAME_0 和 LABEL_NAME_1，例如'Iris-setosa', 'Iris-versicolor'
# 用数据集的本地路径替换 DATASET_PATH ，例如 'iris.data'

import logistic_regression as lr

if __name__ == '__main__':
	logisic_regression = lr.LogisticRegression(
LABEL_NAME_0, LABEL_NAME_1)
	logisic_regression.set_data_from_file(DATASET_PATH)
	logisic_regression.calculate()
	logisic_regression.show()
```

​	或者直接使用python3运行现有的example：

```bash
python example.py
```





## 代码结构：

​	牛顿法的Logistic Regression由文件 *logistic_regression.py* 实现。文件包含一个LogisticRegression类，LogisticRegression包含以下方法：

* init：
  `def __init__(self, label_0, label_1)`
  初始化方法，两个参数为指定的两个label的值。
* append_data：
  `def __append_data(self, data_line)`
  内部方法，处理单行输入。
* set_data_from_file：
  `def set_data_from_file(self, path)`
  读入数据方法，从指定文件读取数据并处理成指定格式。
* calculate：
  `def calculate(self)`
  计算结果方法，实现了牛顿法LogisticRegression。
* show：
  `def show(self, c0='r', c1='b')`
  打印结果方法，绘图。






## 实验结果：




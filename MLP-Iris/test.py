import numpy as np
from MLP_model import MLP
from data_processing import data_read

iris_data = data_read('iris.data')
#test_x = [[6.1],[2.2],[4.9],[1.8]]

def train():
    net = MLP([4, 20, 3])
    net.SGD(iris_data, 20, 0.001)
    return net

def predict(net,test_x):
    predict_label = net.forward(test_x)
    if np.argmax(predict_label) == 0:
        print("It may be Iris-setosa")
    elif np.argmax(predict_label) == 1:
        print("It may be Iris-versicolor")
    else:
        print("It may be Iris-virginica")

net = train()

str = input("Would you want to predict? Please enter 'yes' or 'no': ")
if str == 'yes':
    sepal_length = float(input("plese enter the sepal length:\n"))
    sepal_width = float(input("plese enter the sepal width:\n"))
    petal_length = float(input("plese enter the petal length:\n"))
    petal_width = float(input("plese enter the petal width:\n"))
    test_x = [[sepal_length],[sepal_width],[petal_length],[petal_width]]
    predict(net,test_x)
elif str == 'no':
    print('Over')





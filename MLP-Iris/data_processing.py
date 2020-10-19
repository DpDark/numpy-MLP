import numpy as np
import pandas as pd

def data_read(path):
    df = pd.read_csv(path, sep=',',
                     names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
    labels = np.zeros((df.shape[0], 3))

    #one-hot-encoding
    #Convert character class variables to discrete numeric variables
    for x in range(0, df.shape[0]):
        if (df['class'][x] == 'Iris-setosa'):
            labels[x, 0] = 1
        elif (df['class'][x] == 'Iris-versicolor'):
            labels[x, 1] = 1
        else:
            labels[x, 2] = 1

    values = df.iloc[:, 0:4]
    values = np.array(values)
    data = np.hstack((values, labels))
    return data

def data_shuffle(data):
    np.random.seed(17)
    np.random.shuffle(data)
    X = data[:, :4]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = data[:, 4:7]
    y = y.reshape(y.shape[0], y.shape[1], 1)
    return X, y

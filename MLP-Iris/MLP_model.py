import numpy as np
from numpy import sqrt
from k_fold import get_k_fold_data
from data_processing import data_shuffle

class MLP(object):
    #Initializ parameters
    def __init__(self, layers_node_nums):
        self.num_layers = len(layers_node_nums)
        self.layers_node_nums = layers_node_nums
        self.weights = [np.random.uniform(-sqrt(6 / (m + 1 + n)), sqrt(6 / (m + 1 + n)), (n, m)) for m, n in
                        zip(layers_node_nums[:-1], layers_node_nums[1:])]
        self.biases = [np.random.randn(n, 1) for n in layers_node_nums[1:]]

    #activation funciton
    def tanh(self, a):
        z = (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
        return z

    def tanh_derivatives(self, a):
        return 1 - self.tanh(a) * self.tanh(a)

    def forward(self, x):
        value = x
        for i in range(len(self.weights)):
            value = self.tanh(np.dot(self.weights[i], value) + self.biases[i])
        y = value
        return y

    def backward(self, x, y):

        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        after_activation = [x]
        before_activation = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, after_activation[-1]) + b
            before_activation.append(z)
            activation = self.tanh(z)
            after_activation.append(activation)

        error_rate_grad = after_activation[-1] - y
        delta = error_rate_grad * self.tanh_derivatives(before_activation[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, after_activation[-2].T)
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.tanh_derivatives(before_activation[-l])
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, after_activation[-l - 1].T)
        return grad_b, grad_w

    def grad_updata(self, data, label, leraning_rate):
        b = [np.zeros(b.shape) for b in self.biases]
        w = [np.zeros(w.shape) for w in self.weights]
        grad_b, grad_w = self.backward(data, label)
        b = [b + grad_b for b, grad_b in zip(b, grad_b)]
        w = [w + grad_w for w, grad_w in zip(w, grad_w)]
        self.weights = [w - leraning_rate * grad_w for w, grad_w in zip(self.weights, w)]
        self.biases = [b - leraning_rate * grad_b for b, grad_b in zip(self.biases, b)]

    def SGD(self,all_data,epochs, leraning_rate):
        k = 5
        for j in range(epochs):
            X,y = data_shuffle(all_data)
            acc_sum = 0
            for i in range(k):
                train_data, train_label, test_data, test_label = get_k_fold_data(k, i, X, y)
                for data, label in zip(train_data, train_label):
                    self.grad_updata(data, label, leraning_rate)
                acc_sum = acc_sum + self.evaluate(test_data, test_label)
                average_acc = (acc_sum / k) / len(test_data)
            print("Epoch{0}: accuracy is {1:.3f} % ".format(j + 1, average_acc*100))

    def evaluate(self, data, labels):
        result = 0
        for data, label in zip(data, labels):
            predict_label = self.forward(data)
            if np.argmax(predict_label) == np.argmax(label):
                result += 1
        return result



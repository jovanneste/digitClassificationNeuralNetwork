import pandas as pd
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

dig = load_digits()
onehot_target = pd.get_dummies(dig.target)
x_train, x_test, y_train, y_test = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)

#activation functions
def sigmoid(s):
    return 1/(1+np.exp(-s))
def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)


def sigmoid_derv(s):
    return s*(1-s)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def loss(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss


class NeuralNetwork:
    def __init__(self, x, y):
        self.x = x
        neurons = 128
        self.lr = 0.5
        input_dim = x.shape[1]
        output_dim = y.shape[1]

        self.w1 = np.random.randn(input_dim, neurons)
        self.w2 = np.random.randn(neurons, neurons)
        self.w3 = np.random.randn(neurons, output_dim)

        self.b1 = np.zeros((1, neurons))
        self.b2 = np.zeros((1, neurons))
        self.b3 = np.zeros((1, output_dim))

        self.y = y


    def feedforward():
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)

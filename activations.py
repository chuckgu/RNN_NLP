from __future__ import absolute_import
import theano.tensor as T


def softmax(x):
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)


def softplus(x):
    return T.nnet.softplus(x)


def relu(x):
    return (x + abs(x)) / 2.0

def LeakyReLU(x):
    return (x + abs(x)) / 2.0 + 0.3 * ((x - abs(x)) / 2.0)

def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


def mean(x):
    return T.mean(x)

def max(x):
    return T.max(x)

def linear(x):
    '''
    The function returns the variable that is passed in, so all types work
    '''
    return x
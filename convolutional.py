# -*- coding: utf-8 -*-
from __future__ import absolute_import

import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn

from initializations import glorot_uniform,zero,alloc_zeros_matrix,glorot_normal,numpy_floatX,shared_zeros




class Convolution1D(object):
    def __init__(self, input_dim, nb_filter, filter_length,
                 border_mode='valid', subsample_length=1):

        if border_mode not in {'valid', 'full', 'same'}:
            raise Exception('Invalid border mode for Convolution1D:', border_mode)

        self.nb_filter = nb_filter
        self.input_dim = input_dim
        self.filter_length = filter_length
        self.subsample_length = subsample_length
        self.activation = activations.get(activation)
        self.subsample = (1, subsample_length)
        self.border_mode = border_mode

        self.input = T.tensor3()
        self.x_mask=T.matrix()
        self.W_shape = (nb_filter, input_dim, filter_length, 1)
        self.W = glorot_uniform(self.W_shape)
        self.b = zero((nb_filter,))

        self.params = [self.W, self.b]

    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
        self.x_mask=self.previous.x_mask

    def set_input(self,x):
        self.input=x   

    def set_mask(self,x_mask):
        self.x_mask=x_mask
        
    def get_input(self,train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train)
        else:
            return self.input  

    def get_output(self, train=False):
        X = self.get_input(train)
        X = T.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1)).dimshuffle(0, 2, 1, 3)

        border_mode = self.border_mode
        if border_mode == 'same':
            border_mode = 'full'

        conv_out = T.nnet.conv.conv2d(X, self.W, border_mode=border_mode, subsample=self.subsample)
        if self.border_mode == 'same':
            shift_x = (self.filter_length - 1) // 2
            conv_out = conv_out[:, :, shift_x:X.shape[2] + shift_x, :]

        output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        output = T.reshape(output, (output.shape[0], output.shape[1], output.shape[2])).dimshuffle(0, 2, 1)
        return output



class Convolution2D(Layer):
    def __init__(self, nb_filter, stack_size, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1)):

        if border_mode not in {'valid', 'full', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)

        super(Convolution2D, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = subsample
        self.border_mode = border_mode
        self.nb_filter = nb_filter
        self.stack_size = stack_size

        self.nb_row = nb_row
        self.nb_col = nb_col

        self.input = T.tensor4()
        self.W_shape = (nb_filter, stack_size, nb_row, nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((nb_filter,))

        self.params = [self.W, self.b]



    def get_output(self, train):
        X = self.get_input(train)
        border_mode = self.border_mode
        if dnn.dnn_available() and theano.config.device[:3] == 'gpu':
            if border_mode == 'same':
                assert(self.subsample == (1, 1))
                pad_x = (self.nb_row - self.subsample[0]) // 2
                pad_y = (self.nb_col - self.subsample[1]) // 2
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=(pad_x, pad_y))
            else:
                conv_out = dnn.dnn_conv(img=X,
                                        kerns=self.W,
                                        border_mode=border_mode,
                                        subsample=self.subsample)
        else:
            if border_mode == 'same':
                border_mode = 'full'

            conv_out = T.nnet.conv.conv2d(X, self.W,
                                          border_mode=border_mode,
                                          subsample=self.subsample)
            if self.border_mode == 'same':
                shift_x = (self.nb_row - 1) // 2
                shift_y = (self.nb_col - 1) // 2
                conv_out = conv_out[:, :, shift_x:X.shape[2] + shift_x, shift_y:X.shape[3] + shift_y]

        return self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class MaxPooling1D(Layer):
    def __init__(self, pool_length=2, stride=None, ignore_border=True):
        super(MaxPooling1D, self).__init__()
        self.pool_length = pool_length
        self.stride = stride
        if self.stride:
            self.st = (self.stride, 1)
        else:
            self.st = None

        self.input = T.tensor3()
        self.poolsize = (pool_length, 1)
        self.ignore_border = ignore_border

    def get_output(self, train):
        X = self.get_input(train)
        X = T.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1)).dimshuffle(0, 2, 1, 3)
        output = T.signal.downsample.max_pool_2d(X, ds=self.poolsize, st=self.st, ignore_border=self.ignore_border)
        output = output.dimshuffle(0, 2, 1, 3)
        return T.reshape(output, (output.shape[0], output.shape[1], output.shape[2]))



class MaxPooling2D(Layer):
    def __init__(self, poolsize=(2, 2), stride=None, ignore_border=True):
        super(MaxPooling2D, self).__init__()
        self.input = T.tensor4()
        self.poolsize = poolsize
        self.stride = stride
        self.ignore_border = ignore_border

    def get_output(self, train):
        X = self.get_input(train)
        output = T.signal.downsample.max_pool_2d(X, ds=self.poolsize, st=self.stride, ignore_border=self.ignore_border)
        return output


class UpSample1D(Layer):
    def __init__(self, length=2):
        super(UpSample1D, self).__init__()
        self.length = length
        self.input = T.tensor3()

    def get_output(self, train):
        X = self.get_input(train)
        output = theano.tensor.extra_ops.repeat(X, self.length, axis=1)
        return output



class UpSample2D(Layer):
    def __init__(self, size=(2, 2)):
        super(UpSample2D, self).__init__()
        self.input = T.tensor4()
        self.size = size

    def get_output(self, train):
        X = self.get_input(train)
        Y = theano.tensor.extra_ops.repeat(X, self.size[0], axis=2)
        output = theano.tensor.extra_ops.repeat(Y, self.size[1], axis=3)
        return output


class ZeroPadding2D(Layer):
    def __init__(self, pad=(1, 1)):
        super(ZeroPadding2D, self).__init__()
        self.pad = pad
        self.input = T.tensor4()

    def get_output(self, train):
        X = self.get_input(train)
        pad = self.pad
        in_shape = X.shape
        out_shape = (in_shape[0], in_shape[1], in_shape[2] + 2 * pad[0], in_shape[3] + 2 * pad[1])
        out = T.zeros(out_shape)
        indices = (slice(None), slice(None), slice(pad[0], in_shape[2] + pad[0]), slice(pad[1], in_shape[3] + pad[1]))
        return T.set_subtensor(out[indices], X)

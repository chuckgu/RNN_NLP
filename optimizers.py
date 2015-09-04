from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from initializations import shared_zeros,shared_scalar
from collections import OrderedDict
from six.moves import zip


def SGD(params,cost,mom,lr,decay=1e-6):

    gparams = []
    for param in params:
        gparams.append(T.grad(cost, param))
    # zip just concatenate two lists
    updates = OrderedDict()
    
    #iterations = shared_scalar(0)

    #lr_t = lr * (1.0 / (1.0 + decay * iterations))
    #updates[iterations] = iterations+1.
    #updates[lr]=lr_t


    for param, gparam in zip(params, gparams):
        weight_update = theano.shared(param.get_value(borrow = True) * 0.)
        upd = mom*weight_update - lr * gparam
        updates[weight_update] = upd
        updates[param] = param + upd

    return updates


def RMSprop(params,cost,mom,lr,rho=0.9, epsilon=1e-6):

    gparams = []
    for param in params:
        gparams.append(T.grad(cost, param))
    
    # zip just concatenate two lists
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        acc = theano.shared(param.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * gparam** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        gparam = gparam / gradient_scaling
        
        upd = - lr * gparam
        updates[acc]=acc_new
        updates[param] = param + upd

    return updates  


def Adagrad(params,cost,mom,lr,epsilon=1e-6):

    gparams = []
    for param in params:
        gparams.append(T.grad(cost, param))
    
    # zip just concatenate two lists
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        acc = theano.shared(param.get_value() * 0.)
        acc_new = acc +gparam** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        gparam = gparam / gradient_scaling
        
        upd = - lr * gparam
        updates[acc]=acc_new
        updates[param] = param + upd

    return updates


def Adadelta(params,cost,mom,lr,rho=0.95, epsilon=1e-6):

    gparams = []
    for param in params:
        gparams.append(T.grad(cost, param))
    
    # zip just concatenate two lists
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        acc = theano.shared(param.get_value() * 0.)
        d_acc = theano.shared(param.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * gparam** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        update = gparam * T.sqrt(d_acc + epsilon) / gradient_scaling
        
        upd = - lr * update
        
        new_d_acc = rho * d_acc + (1 - rho) * update ** 2
        
        updates[acc]=acc_new
        updates[d_acc]=new_d_acc
        updates[param] = param + upd

    return updates


def Adam(params,cost,mom,lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8):

    gparams = []
    for param in params:
        gparams.append(T.grad(cost, param))

    # zip just concatenate two lists
    updates = OrderedDict()  
    #lr = shared_scalar(lr)
    iterations = shared_scalar(0)
    updates[iterations] = iterations+1.
    t = iterations + 1
    lr_t = lr * T.sqrt(1-beta_2**t)/(1-beta_1**t)

    for param, gparam in zip(params, gparams):
        
        m = theano.shared(param.get_value() * 0.)  # zero init of moment
        v = theano.shared(param.get_value() * 0.)  # zero init of velocity

        m_t = (beta_1 * m) + (1 - beta_1) * gparam
        v_t = (beta_2 * v) + (1 - beta_2) * (gparam**2)
        p_t = param - lr_t * m_t / (T.sqrt(v_t) + epsilon)
        
        
        updates[m]=m_t
        updates[v]=v_t
        updates[param] = p_t

    return updates  








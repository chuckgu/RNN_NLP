import theano
import theano.tensor as T
import numpy as np
from Initializations import glorot_uniform,zero,alloc_zeros_matrix,glorot_normal,numpy_floatX,orthogonal,one,uniform
import theano.typed_list
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from Activations import relu,LeakyReLU,tanh,sigmoid,linear,mean,max,softmax,hard_sigmoid


def dropout_layer(X, train, trng=RandomStreams(seed=np.random.randint(10e6)),pr=0.5):
    if pr > 0.:
        retain_prob = 1. - pr
        if train:
            X *= trng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        else:
            X *= retain_prob
    return X                


class Layer(object):
    
    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
        self.x_mask=self.previous.x_mask
    
    def set_input(self,x):
        self.input=x

    def get_mask(self):
        return self.x_mask
            
    def set_mask(self,x_mask):
        self.x_mask=x_mask
        
    def get_input(self,train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train)
        else:
            return self.input    

            
            
class Embedding(Layer):
    def __init__(self,n_in,n_hidden,n_out=0,n_out_hidden=0,multi=False,shared_emb=True):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.imatrix()
        self.x_mask=T.imatrix()
        self.multi=multi
        self.W=uniform((n_in,n_hidden))
        self.shared_emb=shared_emb
        if multi:
            self.y= T.imatrix()
            self.y_mask=T.imatrix()
                

        if shared_emb:
            self.params=[self.W]
        else:
            self.n_out=int(n_out)
            self.n_out_hidden=int(n_out_hidden)            
            self.W_multi=uniform((n_out,n_out_hidden))
            self.params=[self.W,self.W_multi]
        
        self.L1 = 0
        self.L2_sqr = 0

    def get_output(self, train=False):
        X = self.get_input(train)
        out = self.W[X]
        return out

    def set_input_y(self,y):
        self.y=y

    def get_input_y(self):
        return self.y

    def set_mask_y(self,y_mask):
        self.y_mask=y_mask

    def get_mask_y(self):
        return self.y_mask

    def get_multi_output(self):
        y = self.y
        if self.shared_emb: out = self.W[y]
        else: out = self.W_multi[y]
        return out


class Activation(Layer):
    '''
        Apply an activation function to an output.
    '''
    def __init__(self, activation, target=0, beta=0.1):
        self.activation = eval(activation)
        self.target = target
        self.beta = beta
        self.params=[]
        
        self.L1 = 0
        self.L2_sqr = 0
 

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.activation(X)

class Drop_out(Layer):
    def __init__(self,pr=0.5):
        self.input= T.tensor3()
        self.x_mask=T.matrix()   
        self.trng = RandomStreams(seed=np.random.randint(10e6))
        self.params=[]
        self.L1=0
        self.L2_sqr=0
        self.pr=pr
        
    
    def get_output(self,train=False):
        X = self.get_input(train)
        if self.pr > 0.:
            retain_prob = 1. - self.pr
            if train:
                X *= self.trng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            else:
                X *= retain_prob
        return X        


class Pool(Layer):
    def __init__(self,mode='mean'):
        self.input= T.tensor3()
        self.x_mask=T.matrix()
        #self.activation=eval(activation)
        self.mode=mode

        self.params=[]
        
        self.L1 = 0
        self.L2_sqr = 0

    
    def get_output(self,train=False):
        if self.mode is 'mean':
            X=self.get_input(train)
            proj = (X * self.x_mask[:, :, None]).sum(axis=0)
            output = proj / self.x_mask.sum(axis=0)[:, None]    
        elif self.mode is 'final':
            X=self.get_input(train)
            proj = (X * self.x_mask[:, :, None])
            output=proj[self.x_mask.sum(axis=0),T.arange(proj.shape[1])]
        return output


class FC_layer(Layer):
    def __init__(self,n_in,n_hidden,activation='linear'):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        self.x_mask=T.matrix()
        self.activation=eval(activation)
        
        
        self.W=glorot_uniform((n_in,n_hidden))
        self.b=zero((n_hidden,))

        
        self.params=[self.W,self.b]
        
        self.L1 = T.sum(abs(self.W))+T.sum(abs(self.b))
        self.L2_sqr = T.sum(self.W**2)+T.sum(self.b**2)
  
    
    def get_output(self,train=False):
        X=self.get_input(train)
        output = self.activation(T.dot(X, self.W) + self.b)
        return output     


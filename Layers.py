import theano
import theano.tensor as T
import numpy as np
from initializations import glorot_uniform,zero,alloc_zeros_matrix,glorot_normal,numpy_floatX
import theano.typed_list
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

SEED = 123
np.random.seed(SEED)

def dropout_layer(state_before, use_noise, trng,pr=0.5):
    proj = T.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=pr, n=1,
                                        dtype=state_before.dtype)),
                         state_before * pr)
    return proj


class drop_out(object):
    def __init__(self,use_dropout,pr=0.5):
        self.input= T.tensor3()
        self.x_mask=T.matrix()   
        self.use_noise = theano.shared(numpy_floatX(0.))
        self.trng = RandomStreams(SEED)
        self.params=[]
        self.use_dropout=use_dropout
        self.L1=0
        self.L2_sqr=0
        self.pr=pr
        
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
    
    def get_output(self,train=False):
        if train:
            if self.use_dropout is True: self.use_noise.set_value(1.)    
        else:self.use_noise.set_value(0.)
        
        self.input=self.get_input(train) 
        
        return dropout_layer(self.input, self.use_noise, self.trng,self.pr)
   

class hidden(object):
    def __init__(self,n_in,n_hidden):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        self.x_mask=T.matrix()
        
        
        self.W_hh=glorot_uniform((n_hidden,n_hidden))
        self.W_in=glorot_uniform((n_in,n_hidden))
        self.bh=zero((n_hidden,))
        
        self.params=[self.W_hh,self.W_in,self.bh]
        
        
        self.L1 = T.sum(abs(self.W_hh))+T.sum(abs(self.W_in))
        self.L2_sqr = T.sum(self.W_hh**2) + T.sum(self.W_in**2)

    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
        self.x_mask=self.previous.x_mask
    
    def _step(self,x_t,x_m, h_tm1):
        h=T.tanh(T.dot(h_tm1, self.W_hh) + T.dot(x_t, self.W_in) + self.bh)
        h=h*x_m[:,None]
        return h
        
    def set_input(self,x):
        self.input=x

    def set_mask(self,x_mask):
        self.x_mask=x_mask
        
    def get_input(self,train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train)
        else:
            return self.input    
    
    def get_output(self,train=False):
        X=self.get_input(train)
        X_mask=self.x_mask
        h, _ = theano.scan(self._step, 
                             sequences = [X,X_mask],
                             outputs_info = alloc_zeros_matrix(self.input.shape[1],
                                                                            self.n_hidden))

        return h
        



class lstm(object):
    def __init__(self,n_in,n_hidden):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        
        
        self.W_i = glorot_uniform((n_in,n_hidden))
        self.U_i = glorot_uniform((n_hidden,n_hidden))
        self.b_i = zero((n_hidden,))

        self.W_f = glorot_uniform((n_in,n_hidden))
        self.U_f = glorot_uniform((n_hidden,n_hidden))
        self.b_f = zero((n_hidden,))

        self.W_c = glorot_uniform((n_in,n_hidden))
        self.U_c = glorot_uniform((n_hidden,n_hidden))
        self.b_c = zero((n_hidden,))

        self.W_o = glorot_uniform((n_in,n_hidden))
        self.U_o = glorot_uniform((n_hidden,n_hidden))
        self.b_o = zero((n_hidden,))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]
        
        self.L1 = T.sum(abs(self.W_i))+T.sum(abs(self.U_i))+\
                  T.sum(abs(self.W_f))+T.sum(abs(self.U_f))+\
                  T.sum(abs(self.W_c))+T.sum(abs(self.U_c))+\
                  T.sum(abs(self.W_o))+T.sum(abs(self.U_o))
        
        self.L2_sqr = T.sum(self.W_i**2) + T.sum(self.U_i**2)+\
                      T.sum(self.W_f**2) + T.sum(self.U_f**2)+\
                      T.sum(self.W_c**2) + T.sum(self.U_c**2)+\
                      T.sum(self.W_o**2) + T.sum(self.U_o**2)
        
    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
        self.x_mask=self.previous.x_mask
    
        
    def _step(self, x_t,x_m, h_tm1, c_tm1): 
        i_t = T.tanh(T.dot(x_t, self.W_i) + self.b_i + T.dot(h_tm1, self.U_i))
        f_t = T.tanh(T.dot(x_t, self.W_f) + self.b_f + T.dot(h_tm1, self.U_f))
        c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_c) + self.b_c + T.dot(h_tm1, self.U_c))
        c_t = x_m[:, None] * c_t + (1. - x_m)[:, None] * c_tm1
        
        o_t = T.tanh( T.dot(x_t, self.W_o) + self.b_o + T.dot(h_tm1, self.U_o))
        h_t = o_t * T.tanh(c_t)

        h_t = x_m[:, None] * h_t + (1. - x_m)[:, None] * h_tm1
        
        return h_t, c_t
       
    def set_input(self,x):
        self.input=x   

    def set_mask(self,x_mask):
        self.x_mask=x_mask
        
    def get_input(self,train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train)
        else:
            return self.input    
    
    def get_output(self,train=False):
        X=self.get_input(train)
        X_mask=self.x_mask        
        [h,c], _ = theano.scan(self._step, 
                             sequences = [X,X_mask],
                             outputs_info = [alloc_zeros_matrix(self.input.shape[1],
                                                                            self.n_hidden),
                                             alloc_zeros_matrix(self.input.shape[1],
                                                                            self.n_hidden)])

        return h

class gru(object):
    def __init__(self,n_in,n_hidden):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.input= T.tensor3()
        
        
        self.W_z = glorot_uniform((n_in,n_hidden))
        self.U_z = glorot_uniform((n_hidden,n_hidden))
        self.b_z = zero((n_hidden,))

        self.W_r = glorot_uniform((n_in,n_hidden))
        self.U_r = glorot_uniform((n_hidden,n_hidden))
        self.b_r = zero((n_hidden,))

        self.W_h = glorot_uniform((n_in,n_hidden)) 
        self.U_h = glorot_uniform((n_hidden,n_hidden))
        self.b_h = zero((n_hidden,))

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        ]

        self.L1 = T.sum(abs(self.W_z))+T.sum(abs(self.U_z))+\
                  T.sum(abs(self.W_r))+T.sum(abs(self.U_r))+\
                  T.sum(abs(self.W_h))+T.sum(abs(self.U_h))
        
        self.L2_sqr = T.sum(self.W_z**2) + T.sum(self.U_z**2)+\
                      T.sum(self.W_r**2) + T.sum(self.U_r**2)+\
                      T.sum(self.W_h**2) + T.sum(self.U_h**2)        
        
    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
        self.x_mask=self.previous.x_mask
    
    def _step(self,x_t,x_m, h_tm1):
        
        z = T.tanh(T.dot(x_t, self.W_z) + self.b_z + T.dot(h_tm1, self.U_z))
        r = T.tanh(T.dot(x_t, self.W_r) + self.b_r + T.dot(h_tm1, self.U_r))
        hh_t = T.tanh(T.dot(x_t, self.W_h) + self.b_h + T.dot(r * h_tm1, self.U_h))
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t=x_m[:,None] * h_t + (1. - x_m)[:,None] * h_tm1
        
        return h_t
        
    def set_input(self,x):
        self.input=x

    def set_mask(self,x_mask):
        self.x_mask=x_mask
        
    def get_input(self,train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train)
        else:
            return self.input    
    
    def get_output(self,train=False):
        X=self.get_input(train)
        X_mask=self.x_mask
        h, _ = theano.scan(self._step, 
                             sequences = [X,X_mask],
                             outputs_info = alloc_zeros_matrix(self.input.shape[1],
                                                                            self.n_hidden))

        return h




class BiDirectionLSTM(object):
    def __init__(self,n_in,n_hidden,output_mode='concat'):
        self.n_in=int(n_in)
        if output_mode is 'concat': n_hidden=int(n_hidden/2)
        self.n_hidden=int(n_hidden)
        self.output_mode = output_mode
        self.input= T.tensor3()
        self.x_mask=T.matrix()
        
        # forward weights
        self.W_i = glorot_uniform((n_in,n_hidden))
        self.U_i = glorot_uniform((n_hidden,n_hidden))
        self.b_i = zero((n_hidden,))

        self.W_f = glorot_uniform((n_in,n_hidden))
        self.U_f = glorot_uniform((n_hidden,n_hidden))
        self.b_f = zero((n_hidden,))

        self.W_c = glorot_uniform((n_in,n_hidden))
        self.U_c = glorot_uniform((n_hidden,n_hidden))
        self.b_c = zero((n_hidden,))

        self.W_o = glorot_uniform((n_in,n_hidden))
        self.U_o = glorot_uniform((n_hidden,n_hidden))
        self.b_o = zero((n_hidden,))
        
        # backward weights
        self.Wb_i = glorot_uniform((n_in,n_hidden))
        self.Ub_i = glorot_uniform((n_hidden,n_hidden))
        self.bb_i = zero((n_hidden,))

        self.Wb_f = glorot_uniform((n_in,n_hidden))
        self.Ub_f = glorot_uniform((n_hidden,n_hidden))
        self.bb_f = zero((n_hidden,))

        self.Wb_c = glorot_uniform((n_in,n_hidden))
        self.Ub_c = glorot_uniform((n_hidden,n_hidden))
        self.bb_c = zero((n_hidden,))
        
        self.Wb_o = glorot_uniform((n_in,n_hidden))
        self.Ub_o = glorot_uniform((n_hidden,n_hidden))
        self.bb_o = zero((n_hidden,))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,

            self.Wb_i, self.Ub_i, self.bb_i,
            self.Wb_c, self.Ub_c, self.bb_c,
            self.Wb_f, self.Ub_f, self.bb_f,
            self.Wb_o, self.Ub_o, self.bb_o,
        ]

        self.L1 = T.sum(abs(self.W_i))+T.sum(abs(self.U_i))+\
                  T.sum(abs(self.W_f))+T.sum(abs(self.U_f))+\
                  T.sum(abs(self.W_c))+T.sum(abs(self.U_c))+\
                  T.sum(abs(self.W_o))+T.sum(abs(self.U_o))+\
                  T.sum(abs(self.Wb_i))+T.sum(abs(self.Ub_i))+\
                  T.sum(abs(self.Wb_f))+T.sum(abs(self.Ub_f))+\
                  T.sum(abs(self.Wb_c))+T.sum(abs(self.Ub_c))+\
                  T.sum(abs(self.Wb_o))+T.sum(abs(self.Ub_o))
        
        self.L2_sqr = T.sum(self.W_i**2) + T.sum(self.U_i**2)+\
                      T.sum(self.W_f**2) + T.sum(self.U_f**2)+\
                      T.sum(self.W_c**2) + T.sum(self.U_c**2)+\
                      T.sum(self.W_o**2) + T.sum(self.U_o**2)+\
                      T.sum(self.Wb_i**2) + T.sum(self.Ub_i**2)+\
                      T.sum(self.Wb_f**2) + T.sum(self.Ub_f**2)+\
                      T.sum(self.Wb_c**2) + T.sum(self.Ub_c**2)+\
                      T.sum(self.Wb_o**2) + T.sum(self.Ub_o**2)

        
        
    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
        self.x_mask=self.previous.x_mask
    
        
    def _fstep(self, x_t,x_m, h_tm1, c_tm1): 
        i_t = T.tanh(T.dot(x_t, self.W_i) + self.b_i + T.dot(h_tm1, self.U_i))
        f_t = T.tanh(T.dot(x_t, self.W_f) + self.b_f + T.dot(h_tm1, self.U_f))
        c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_c) + self.b_c + T.dot(h_tm1, self.U_c))
        c_t = x_m[:, None] * c_t + (1. - x_m)[:, None] * c_tm1        
        
        o_t = T.tanh( T.dot(x_t, self.W_o) + self.b_o + T.dot(h_tm1, self.U_o))
        h_t = o_t * T.tanh(c_t)
        h_t = x_m[:, None] * h_t + (1. - x_m)[:, None] * h_tm1
        
        return h_t, c_t


    def _bstep(self, x_t,x_m, h_tm1, c_tm1): 
        i_t = T.tanh(T.dot(x_t, self.Wb_i) + self.bb_i + T.dot(h_tm1, self.Ub_i))
        f_t = T.tanh(T.dot(x_t, self.Wb_f) + self.bb_f + T.dot(h_tm1, self.Ub_f))
        c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.Wb_c) + self.bb_c + T.dot(h_tm1, self.Ub_c))
        c_t = x_m[:, None] * c_t + (1. - x_m)[:, None] * c_tm1            
        
        o_t = T.tanh( T.dot(x_t, self.Wb_o) + self.bb_o + T.dot(h_tm1, self.Ub_o))
        h_t = o_t * T.tanh(c_t)
        h_t = x_m[:, None] * h_t + (1. - x_m)[:, None] * h_tm1        
        
        return h_t, c_t        
       
    def set_input(self,x):
        self.input=x   
        
    def set_mask(self,x_mask):
        self.x_mask=x_mask
        
    def get_input(self,train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train)
        else:
            return self.input    
    
    def get_forward_output(self,train=False):
        X=self.get_input(train)
        X_mask=self.x_mask
        [h,c], _ = theano.scan(self._fstep, 
                             sequences = [X,X_mask],
                             outputs_info = [alloc_zeros_matrix(self.input.shape[1],
                                                                            self.n_hidden),
                                             alloc_zeros_matrix(self.input.shape[1],
                                                                            self.n_hidden)])

        return h
        
    def get_backward_output(self,train=False):
        X=self.get_input(train)
        X_mask=self.x_mask
        [h,c], _ = theano.scan(self._bstep, 
                             sequences = [X,X_mask],
                             outputs_info = [alloc_zeros_matrix(self.input.shape[1],
                                                                            self.n_hidden),
                                             alloc_zeros_matrix(self.input.shape[1],
                                                                            self.n_hidden)],
                                            go_backwards = True)

        return h  


    def get_output(self,train=False):
        forward = self.get_forward_output(train)
        backward = self.get_backward_output(train)
        if self.output_mode is 'sum':
            return forward + backward
        elif self.output_mode is 'concat':
            return T.concatenate([forward, backward], axis=2)
        else:
            raise Exception('output mode is not sum or concat')


class BiDirectionGRU(object):
    def __init__(self,n_in,n_hidden,output_mode='concat'):
        self.n_in=int(n_in)
        if output_mode is 'concat':n_hidden=int(n_hidden/2)
        self.n_hidden=int(n_hidden)
        self.output_mode = output_mode
        self.input= T.tensor3()
        self.x_mask=T.matrix()
        
        # forward weights
        self.W_z = glorot_uniform((n_in,n_hidden))
        self.U_z = glorot_uniform((n_hidden,n_hidden))
        self.b_z = zero((n_hidden,))

        self.W_r = glorot_uniform((n_in,n_hidden))
        self.U_r = glorot_uniform((n_hidden,n_hidden))
        self.b_r = zero((n_hidden,))

        self.W_h = glorot_uniform((n_in,n_hidden)) 
        self.U_h = glorot_uniform((n_hidden,n_hidden))
        self.b_h = zero((n_hidden,))
        
        # backward weights
        self.Wb_z = glorot_uniform((n_in,n_hidden))
        self.Ub_z = glorot_uniform((n_hidden,n_hidden))
        self.bb_z = zero((n_hidden,))

        self.Wb_r = glorot_uniform((n_in,n_hidden))
        self.Ub_r = glorot_uniform((n_hidden,n_hidden))
        self.bb_r = zero((n_hidden,))

        self.Wb_h = glorot_uniform((n_in,n_hidden)) 
        self.Ub_h = glorot_uniform((n_hidden,n_hidden))
        self.bb_h = zero((n_hidden,))        

        self.params = [
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,

            self.Wb_z, self.Ub_z, self.bb_z,
            self.Wb_r, self.Ub_r, self.bb_r,
            self.Wb_h, self.Ub_h, self.bb_h
        ]

        self.L1 = T.sum(abs(self.W_z))+T.sum(abs(self.U_z))+\
                  T.sum(abs(self.W_r))+T.sum(abs(self.U_r))+\
                  T.sum(abs(self.W_h))+T.sum(abs(self.U_h))+\
                  T.sum(abs(self.Wb_z))+T.sum(abs(self.Ub_z))+\
                  T.sum(abs(self.Wb_r))+T.sum(abs(self.Ub_r))+\
                  T.sum(abs(self.Wb_h))+T.sum(abs(self.Ub_h))
        
        self.L2_sqr = T.sum(self.W_z**2) + T.sum(self.U_z**2)+\
                      T.sum(self.W_r**2) + T.sum(self.U_r**2)+\
                      T.sum(self.W_h**2) + T.sum(self.U_h**2)+\
                      T.sum(self.Wb_z**2) + T.sum(self.Ub_z**2)+\
                      T.sum(self.Wb_r**2) + T.sum(self.Ub_r**2)+\
                      T.sum(self.Wb_h**2) + T.sum(self.Ub_h**2)

        
        
    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
        self.x_mask=self.previous.x_mask
    
        
    def _fstep(self,x_t,x_m, h_tm1):
        
        z = T.tanh(T.dot(x_t, self.W_z) + self.b_z + T.dot(h_tm1, self.U_z))
        r = T.tanh(T.dot(x_t, self.W_r) + self.b_r + T.dot(h_tm1, self.U_r))
        hh_t = T.tanh(T.dot(x_t, self.W_h) + self.b_h + T.dot(r * h_tm1, self.U_h))
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t=x_m[:,None] * h_t + (1. - x_m)[:,None] * h_tm1
        
        return h_t


    def _bstep(self,x_t,x_m, h_tm1):
        
        z = T.tanh(T.dot(x_t, self.Wb_z) + self.bb_z + T.dot(h_tm1, self.Ub_z))
        r = T.tanh(T.dot(x_t, self.Wb_r) + self.bb_r + T.dot(h_tm1, self.Ub_r))
        hh_t = T.tanh(T.dot(x_t, self.Wb_h) + self.bb_h + T.dot(r * h_tm1, self.Ub_h))
        h_t = z * h_tm1 + (1 - z) * hh_t
        h_t=x_m[:,None] * h_t + (1. - x_m)[:,None] * h_tm1
        
        return h_t  
       
    def set_input(self,x):
        self.input=x   
        
    def set_mask(self,x_mask):
        self.x_mask=x_mask
        
    def get_input(self,train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train)
        else:
            return self.input    
    
    def get_forward_output(self,train=False):
        X=self.get_input(train)
        mask_x=self.x_mask
        h, _ = theano.scan(self._fstep, 
                             sequences = [X,mask_x],
                             outputs_info = alloc_zeros_matrix(self.input.shape[1],
                                                                            self.n_hidden))

        return h
        
    def get_backward_output(self,train=False):
        X=self.get_input(train)
        mask_x=self.x_mask
        h, _ = theano.scan(self._bstep, 
                             sequences = [X,mask_x],
                             outputs_info = alloc_zeros_matrix(self.input.shape[1],
                                                                            self.n_hidden),
                                            go_backwards = True)

        return h  


    def get_output(self,train=False):
        forward = self.get_forward_output(train)
        backward = self.get_backward_output(train)
        if self.output_mode is 'sum':
            return forward + backward
        elif self.output_mode is 'concat':
            return T.concatenate([forward, backward], axis=2)
        else:
            raise Exception('output mode is not sum or concat')

class decoder(object):
    def __init__(self,n_in,n_hidden,n_out):
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.n_out=int(n_out)
        self.input= T.tensor3()
        self.output= T.tensor3()
        self.x_mask=T.matrix()
        #self.y_mask=T.matrix()

        
        self.W_z = glorot_normal((n_out,n_hidden))
        self.U_z = glorot_normal((n_hidden,n_hidden))
        self.b_z = zero((n_hidden,))

        self.W_r = glorot_normal((n_out,n_hidden))
        self.U_r = glorot_normal((n_hidden,n_hidden))
        self.b_r = zero((n_hidden,))

        self.W_h = glorot_normal((n_out,n_hidden)) 
        self.U_h = glorot_normal((n_hidden,n_hidden))
        self.b_h = zero((n_hidden,))
        
        self.U_att= glorot_normal((self.n_in,1)) 
        self.b_att= zero((1,))

        self.W_yc=glorot_normal((self.n_out,))
        

        self.W_cy = glorot_normal((self.n_in,self.n_hidden))
        self.W_cs= glorot_normal((self.n_in,self.n_hidden))

        
        self.W_ha = glorot_normal((self.n_in,self.n_in))
        self.W_sa= glorot_normal((self.n_hidden,self.n_in))
        

        
        self.W_cl= glorot_normal((self.n_in,self.n_out))
        self.W_yl= glorot_normal((self.n_out,self.n_out))
        self.W_hl= glorot_normal((self.n_hidden,self.n_out))
        
        self.params=[self.W_z,self.U_z,self.b_z,self.W_r,self.U_r,self.b_r,
                   self.W_h,self.U_h,self.b_h,self.W_cy,self.W_cs,self.W_ha,self.W_sa
                     ,self.W_cl,self.W_yl,self.W_hl,self.U_att,self.b_att]
        

        self.L1 = T.sum(abs(self.W_z))+T.sum(abs(self.U_z))+\
                  T.sum(abs(self.W_r))+T.sum(abs(self.U_r))+\
                  T.sum(abs(self.W_h))+T.sum(abs(self.U_h))+\
                  T.sum(abs(self.W_cy))+T.sum(abs(self.W_cs))+\
                  T.sum(abs(self.W_ha))+T.sum(abs(self.W_sa))+\
                  T.sum(abs(self.W_cl))+T.sum(abs(self.W_yl))+\
                  T.sum(abs(self.W_hl))+T.sum(abs(self.U_att))
        self.L2_sqr = T.sum(self.W_z**2) + T.sum(self.U_z**2)+\
                      T.sum(self.W_r**2) + T.sum(self.U_r**2)+\
                      T.sum(self.W_h**2) + T.sum(self.U_h**2)+\
                      T.sum(self.W_cy**2) + T.sum(self.W_cs**2)+\
                      T.sum(self.W_ha**2) + T.sum(self.W_sa**2)+\
                      T.sum(self.W_cl**2) + T.sum(self.W_yl**2)+\
                      T.sum(self.W_hl**2) + T.sum(self.U_att**2)
        
    def _step(self,y_tm1,y_m,s_tm1,h,x_m):
                
        # attention

        pctx__=T.dot(h,self.W_ha)+T.dot(s_tm1,self.W_sa)[None,:,:]
        
        pctx__=T.tanh(pctx__)
        
        e=T.dot(pctx__,self.U_att)+self.b_att
        
        e=T.exp(e.reshape((e.shape[0],e.shape[1])))
        
        e=e/e.sum(0, keepdims=True)
        
        e=e*x_m
  
        c=(h*e[:,:,None]).sum(0)
        

        z = T.nnet.sigmoid(T.dot(y_tm1, self.W_z) + self.b_z + T.dot(s_tm1, self.U_z)+T.dot(c,self.W_cs))
        r = T.nnet.sigmoid(T.dot(y_tm1, self.W_r) + self.b_r + T.dot(s_tm1, self.U_r)+T.dot(c,self.W_cs))
        hh_t = T.tanh(T.dot(y_tm1, self.W_h) + self.b_h + T.dot(r * s_tm1, self.U_h)+T.dot(c,self.W_cy))
        s_t = z * s_tm1 + (1 - z) * hh_t
        
        s_t = (1. - y_m)[:,None] * s_tm1 + y_m[:,None] * s_t
        
        logit=T.tanh(T.dot(s_t, self.W_hl)+T.dot(y_tm1, self.W_yl)+T.dot(c, self.W_cl))
        
        return T.cast(s_t,dtype =theano.config.floatX),T.cast(logit,dtype =theano.config.floatX)  

    def set_previous(self,layer):
        self.previous = layer
        self.input=self.get_input()
        self.x_mask=self.previous.x_mask
        
    def set_input(self,x):
        self.input=x

        
    def get_input(self,train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output(train)
        else:
            return self.input    



    def get_sample(self,y,s_tm1):
        c=self.get_input()
        X_mask=self.previous.x_mask
        y_mask=T.alloc(1.,y.shape[0])
        
        h,logit=self._step(y,y_mask,s_tm1,c,X_mask) 
        

        return h,logit
    
    def get_output(self,y,y_mask,init_state,train=False):
        X=self.get_input(train)  
        X_mask=self.previous.x_mask
  
        ### shift 1 sequence backward
        y_shifted=T.zeros_like(y)
        y_shifted=T.set_subtensor(y_shifted[1:],y[:-1])
        y=y_shifted 

        ### shift 1 sequence backward
        y_shifted=T.zeros_like(y_mask)
        y_shifted=T.set_subtensor(y_shifted[1:],y_mask[:-1])
        y_mask=y_shifted 
        
        [h,logit], _ = theano.scan(self._step, 
                                     sequences = [y,y_mask],
                                     outputs_info = [init_state,
                                                     None],
                                     non_sequences=[X,X_mask])

        return logit

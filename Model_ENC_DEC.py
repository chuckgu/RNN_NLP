import theano
import theano.tensor as T
import numpy as np
import copy
import os
import datetime
import cPickle as pickle
from Loss import nll_multiclass_3d,categorical_crossentropy
from Initializations import glorot_uniform,zero,alloc_zeros_matrix,norm_weight,glorot_normal
from Utils import Progbar,ndim_tensor,make_batches,slice_X,seq_to_text
from Optimizers import SGD,RMSprop,Adagrad,Adadelta,Adam
from Layers import dropout_layer
import Callbacks as cbks


mode = theano.Mode(linker='cvm', optimizer='fast_run') #the runtime algo to execute the code is in c

class ENC_DEC(object):
    
    def __init__(self,n_in,n_hidden,n_decoder,n_out,
                 n_epochs=400,n_batch=16,maxlen=20,n_words_x=10000,n_words_y=10000,dim_word=100,
                 snapshot=100,sample_Freq=100,val_Freq=100,shared_emb=False,L1_reg=0,L2_reg=0):
        
        self.n_in=int(n_in)
        self.n_hidden=int(n_hidden)
        self.n_decoder=int(n_decoder)
        self.n_out=int(n_out)
        
        self.n_batch=int(n_batch)
        self.shared_emb=shared_emb
    
        
        self.n_epochs=n_epochs
        self.maxlen= int(maxlen)   
        self.dim_word=dim_word
        self.n_words_x=n_words_x
        self.n_words_y=n_words_y
        
        self.x = T.matrix(name = 'x', dtype = 'int32')
        self.y = T.matrix(name = 'y', dtype = 'int32')
        
        self.x_mask = T.matrix(name = 'x_mask', dtype = 'float32')
        self.y_mask = T.matrix(name = 'y_mask', dtype = 'float32')
        
        self.x_emb = T.tensor3(name = 'x', dtype = 'float32')
        self.y_emb = T.tensor3(name = 'y', dtype = 'float32')        
        
        self.W_hy = glorot_uniform((self.n_out,self.n_words_y))
        self.b_hy = zero((self.n_words_y,))
        
        self.W_hi = glorot_uniform((self.n_hidden,self.n_decoder))
        self.b_hi = zero((n_decoder,))
        
        self.Wemb=glorot_normal((self.n_words_x,self.dim_word))
        self.x_emb=self.Wemb[self.x]
        self.y_emb=self.Wemb[self.y]
        if not self.shared_emb: 
            self.Wemb_dec=glorot_normal((self.n_words_y,self.dim_word))
            self.y_emb=self.Wemb_dec[self.y]
         
        self.layers = []
        self.params=[]
        self.errors=[]
        self.val_errors=[]
        
        self.snapshot=int(snapshot)
        self.sample_Freq=int(sample_Freq)
        self.val_Freq=int(val_Freq)
       
        self.L1_reg=L1_reg
        self.L2_reg=L2_reg    
        self.L1= 0
        self.L2_sqr= 0
    

        
    def add(self,layer): 
  
        self.layers.append(layer)
    
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
        else:
            self.set_input()
            self.set_mask()
  
        self.params+=layer.params
        self.L1 += layer.L1
        self.L2_sqr += layer.L2_sqr


    def set_params(self,**params):
        return
    
    def __getstate__(self):
        """ Return state sequence."""
        params = self.params  # parameters set in constructor
        weights = [p.get_value() for p in self.params]
        lr=self.lr
        error=self.errors
        state = (params, weights,lr,error)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)

        for param in self.params:
            param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        params, weights, lr,error = state
        #self.set_params(**params)
        #self.ready()
        self._set_weights(weights)
        self.lr=lr
        self.errors=error

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        print("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """
        print("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()       

    def set_mask(self):

        self.layers[0].x_mask = self.x_mask        
    
    def set_input(self):
        for l in self.layers:
            if hasattr(l, 'input'):
                self.layers[0].input = self.x_emb
                break

    def get_input(self,train):
        if not hasattr(self.layers[0], 'input'):
            self.set_input()
        return self.layers[0].get_input(train)  

    def get_output(self,train):
        
        ## calculate initial state
        ctx=self.layers[-1].get_input(train)
        ctx_mean = (ctx * self.x_mask[:,:,None]).sum(0) / self.x_mask.sum(0)[:,None]
        ctx_mean = dropout_layer(ctx_mean, train)
        init_state=T.tanh(T.dot(ctx_mean, self.W_hi) + self.b_hi)
        
        proj=self.layers[-1].get_output(self.y_emb,self.y_mask,init_state,train)
        
        ### fianl prediction formular
                                                        
        proj = dropout_layer(proj, train)                                              
        self.y_pred = T.dot(proj, self.W_hy) + self.b_hy
        
        y_p_m = T.reshape(self.y_pred, (self.y_pred.shape[0] * self.y_pred.shape[1], -1))
        y_p_s = T.nnet.softmax(y_p_m)
        #p_y_given_x = T.reshape(y_p_s, self.y_pred.shape)
        
        return T.reshape(y_p_s, self.y_pred.shape)



    def build(self):

        ### set up parameters
    
        if self.shared_emb: self.params+=[self.W_hi, self.b_hi,self.W_hy, self.b_hy, self.Wemb]
        else: self.params+=[self.W_hi, self.b_hi,self.W_hy, self.b_hy, self.Wemb, self.Wemb_dec]

        ### set up regularizer                               
   
        self.L1 += T.sum(abs(self.W_hy))    
        self.L2_sqr += T.sum(self.W_hy**2)
                                                                  

    def compile(self,optimizer='Adam',loss='nll_multiclass_3d'):      
        
        self.build()        

        next_y=T.matrix()
        next_h=T.matrix()
        
        # output of model
        self.y_train = self.get_output(train=True)
        self.y_test = self.get_output(train=False)
        y_pred=self.get_sample(next_y,next_h)

        

        if type(self.x) == list:
            train_ins = self.x + self.x_mask +  [self.y,self.y_mask]
            test_ins = self.x + self.x_mask  + [self.y,self.y_mask]
            predict_ins = self.x + self.x_mask+ next_y + next_h
        else:
            train_ins = [self.x, self.x_mask, self.y, self.y_mask]
            test_ins = [self.x, self.x_mask, self.y, self.y_mask]
            predict_ins = [self.x, self.x_mask, next_y, next_h]
 
        ### cost and updates    
 
        self.loss = eval(loss)

        train_loss=self.loss(self.y, self.y_mask ,self.y_train)
        test_loss=self.loss(self.y, self.y_mask ,self.y_test)

        train_loss.name = 'train_loss'
        test_loss.name = 'test_loss'
        
        #train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)))
        #test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)))
        #train_accuracy = T.mean(T.eq(self.y, T.argmax(self.y_train, axis=-1)))
        #test_accuracy = T.mean(T.eq(self.y, T.argmax(self.y_test, axis=-1)))
        
        cost = train_loss +self.L2_reg * self.L2_sqr
        
        self.optimizer=eval(optimizer)()
        updates=self.optimizer.get_updates(self.params,train_loss)    

        print 'Optimizer: '+optimizer   

        ### compile theano functions 

        idx=T.lscalar() 
        if self.shared_emb:                                    
            self.get_embedding = theano.function(inputs = [idx,],
                                         outputs = self.Wemb[idx],
                                         mode = mode) 
        else:
            self.get_embedding = theano.function(inputs = [idx,],
                                         outputs = self.Wemb_dec[idx],
                                         mode = mode)                                              

        self._train = theano.function(inputs = train_ins,
                                      outputs = cost,
                                      updates = updates,
                                      mode = mode)
                    
        self._test = theano.function(inputs = test_ins,
                                              outputs = test_loss,
                                              mode = mode) 
        
        self._predict = theano.function(inputs = predict_ins,
                                             outputs = y_pred,
                                             mode = mode)        
     
 

    def train(self,train_set,val_set,worddict,verbose,shuffle=True,show_accuracy=False):

        train_set_x = np.asarray(train_set[0], dtype='int32')
        mask_set_x = np.asarray(train_set[1], dtype='float32')
        train_set_y = np.asarray(train_set[2], dtype='int32')
        mask_set_y = np.asarray(train_set[3], dtype='float32')
        
        ins = [train_set_x, mask_set_x, train_set_y,mask_set_y]

        val_set_x = np.asarray(val_set[0], dtype='int32')
        mask_val_set_x = np.asarray(val_set[1], dtype='float32')        
        val_set_y = np.asarray(val_set[2], dtype='int32')   
        mask_val_set_y = np.asarray(val_set[3], dtype='float32')             
        
        val_ins = [val_set_x, mask_val_set_x, val_set_y,mask_val_set_y]
        
                            
        ###############
        # TRAIN MODEL #
        ###############
        print 'Training model ...'

        nb_train_sample = train_set_x.shape[1]
        index_array = np.arange(nb_train_sample)

        ### call back###
        history = cbks.History()
        
        callbacks = [history, cbks.BaseLogger()]

        callbacks = cbks.CallbackList(callbacks)
        
        if show_accuracy:
            f = self._train_with_acc
            out_labels = ['loss', 'acc']
            metrics = ['loss', 'acc', 'val_loss', 'val_acc']
        else:
            f = self._train
            out_labels = ['loss']
            metrics = ['loss', 'val_loss']
            
        
      
        do_validation = True

        callbacks._set_model(self)
        callbacks._set_params({
            'batch_size': self.n_batch,
            'nb_epoch': self.n_epochs,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': metrics,
        })
        callbacks.on_train_begin()        
        
        for epoch in range(self.n_epochs):
            callbacks.on_epoch_begin(epoch)
            
            if shuffle: np.random.shuffle(index_array)

            train_losses=[]

            batches = make_batches(nb_train_sample, self.n_batch)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
            
                batch_ids = index_array[batch_start:batch_end]
                try:
                    ins_batch = slice_X(ins, batch_ids)
                except TypeError as err:
                    print('TypeError while preparing batch. \
                        If using HDF5 input data, pass shuffle="batch".\n')
                    raise

                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                        
                cost = f(*ins_batch) 

                if np.isnan(cost) or np.isinf(cost):
                    raise ValueError('NaN detected')

                                   
                train_losses.append(cost) 
                #train_batch_sizes.append(get_batch_size(idx, n_train))    
                if type(cost) != list:
                    cost = [cost]
                    
                for l, o in zip(out_labels, cost):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)                         
                            
            epoch_logs = {} 
            # compute loss on validation set
            if np.mod(epoch+1,self.val_Freq)==0:

                val_outs = self._test_loop(self._test, val_ins, batch_size=self.n_batch, verbose=0)
                self.val_errors.append(val_outs)
                
                if type(val_outs) != list:
                    val_outs = [val_outs]                
                
                for l, o in zip(out_labels, val_outs):
                    epoch_logs['val_' + l] = o
                    
            callbacks.on_epoch_end(epoch, epoch_logs)                    
      
            this_train_loss = np.average(train_losses)      
            self.errors.append(this_train_loss)
            
            ### generating sample.. 
            if np.mod(epoch+1,self.sample_Freq)==0:
               # self.set_train(False) 
                print 'Generating a sample...'               
                
                rand=np.random.randint(1,nb_train_sample)
                
                test=train_set_x[:,rand][:,None]
                mask=mask_set_x[:,rand][:,None]
                truth=train_set_y[:,rand][:,None]  
                
                ins_gen=[test,mask,truth]
                
                self.generate(ins_gen,worddict)
                          
            ### autimatically saving snapshot ..
            if np.mod(epoch+1,self.snapshot)==0:
                if epoch is not self.n_epochs: self.save()            


    def generate(self, test_set, worddict, with_truth=True,batch_size=1):
        
        test_set_x = np.asarray(test_set[0], dtype='int32')
        mask_set_x = np.asarray(test_set[1], dtype='float32')     
        ins = [test_set_x, mask_set_x]        

        if with_truth:
            test_set_y = np.asarray(test_set[2], dtype='int32')
            ins_t=[test_set_y]
        
        nb_sample = ins[0].shape[1]
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids)
            ins_batch_t = slice_X(ins_t, batch_ids)

            batch_outs = self._generate_loop(self._predict, ins_batch)
            print 'Input: ',seq_to_text(ins_batch[0],worddict)  
            if with_truth: print 'Truth: ',seq_to_text(ins_batch_t[0],worddict)
            print 'Sample: ',seq_to_text(batch_outs[1],worddict)




                    
    def get_sample(self,y,h):
        
        ctx=self.layers[-1].get_input(False)
        ctx_mean = (ctx * self.x_mask[:,:,None]).sum(0) / self.x_mask.sum(0)[:,None]
        
        h = T.switch(h[0] < 0, 
                        T.tanh(T.dot(ctx_mean, self.W_hi) + self.b_hi), 
                        h) 
        
        h,logit=self.layers[-1].get_sample(y,h)
        logit = dropout_layer(logit, False)
        y_gen = T.dot(logit, self.W_hy) + self.b_hy
            
        p_y_given_x_gen=T.nnet.softmax(y_gen)
            
        return h,logit,p_y_given_x_gen                            
   

                                      
    def _generate_loop(self,f,ins,stochastic=True,k=3):
        
       # X_test=np.asarray(X_test[:,None],dtype='int32')
        #X_mask=np.asarray(X_mask[:,None],dtype='float32')
        
        X_test=ins[0]
        X_mask=ins[1]
        
        if X_test.ndim==1: X_test=X_test[:,None]
        if X_mask.ndim==1: X_mask=X_mask[:,None]

        sample=[]
        sample_proba=[]
        
        sample_score = []
        
        live_k = 1
        dead_k = 0
        
        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype('float32')
        hyp_states = []

        next_w=np.zeros((1,self.n_out)).astype('float32') 
        h_w=-1*np.ones((1,self.n_decoder)).astype('float32')
        
        for i in xrange(self.maxlen):
            gen_ins=[X_test,X_mask,next_w,h_w]
            h_w,logit,p_y_given_x_gen=f(*gen_ins)
            sample_proba.append(p_y_given_x_gen.flatten())
            
            if stochastic: ### stochastic sampling
                
                result = np.argmax(p_y_given_x_gen, axis = -1)[0] 
                

                sample.append(result) 
                
                w=self.get_embedding(result)
                
                next_w=np.asarray(w.reshape((1,self.n_out))).astype('float32')

             
            else:   
                p_y_given_x_gen=np.array(p_y_given_x_gen).astype('float32')

                #print p_y_given_x_gen
                cand_scores = hyp_scores[:,None] - np.log(p_y_given_x_gen.flatten())
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k-dead_k)]
                
                voc_size = p_y_given_x_gen.shape[1]
                trans_indices = ranks_flat / voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]
    
                new_hyp_samples = []
                new_hyp_scores = np.zeros(k-dead_k).astype('float32')
               # new_hyp_states = []

                
                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    
                    new_hyp_samples.append(hyp_samples[ti]+[wi])
                    new_hyp_scores[idx] = copy.copy(costs[ti])
                   # new_hyp_states.append(copy.copy(result[ti]))
                
                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                #hyp_states = []
    
                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        #hyp_states.append(new_hyp_states[idx])
                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k
    
                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break
    
                next_w = np.array([w[-1] for w in hyp_samples])
                
                w=self.get_embedding(next_w[0])
                
                next_w=np.asarray(w.reshape((1,self.n_out))).astype('float32')
                #next_state = np.array(hyp_states)
        
        if not stochastic:
        # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])
            sample=sample[np.argmin(sample_score)]        
        

        return sample_proba,sample
   
    def evaluate(self, test_set, batch_size=128, show_accuracy=False, verbose=1):
        
        test_set_x = np.asarray(test_set[0], dtype='int32')
        mask_set_x = np.asarray(test_set[1], dtype='float32')
        test_set_y = np.asarray(test_set[2], dtype='int32')
        mask_set_y = np.asarray(test_set[3], dtype='float32')
        
        ins = [test_set_x, mask_set_x, test_set_y,mask_set_y]        
        
        if show_accuracy:
            f = self._test_with_acc
        else:
            f = self._test
        outs = self._test_loop(f, ins, batch_size, verbose)
        if show_accuracy:
            return outs
        else:
            return outs[0]

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        '''
            Abstract method to loop over some data in batches.
        '''
        nb_sample = ins[0].shape[1]
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            ins_batch = slice_X(ins, batch_ids)

            batch_outs = f(*ins_batch)
            if type(batch_outs) == list:
                if batch_index == 0:
                    for batch_out in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    outs[i] += batch_out * len(batch_ids)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += batch_outs * len(batch_ids)

            if verbose == 1:
                progbar.update(batch_end)
        for i, out in enumerate(outs):
            outs[i] /= nb_sample
        return outs        
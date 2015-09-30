import numpy as np
import matplotlib.pyplot as plt
import os
from Layers import Drop_out,Embedding,FC_layer,Pool,Activation
from Recurrent_Layers import Hidden,LSTM,GRU,BiDirectionLSTM,Decoder,BiDirectionGRU
from Model import RNN
from Load_data import load_data,prepare_full_data
from Utils import Progbar
from sklearn.metrics import accuracy_score


#theano.config.exception_verbosity='high'

#theano.config.optimizer='None' 

n_epochs = 100
optimizer="Adam"
loss='nll_multiclass'
#RMSprop,SGD,Adagrad,Adadelta,Adam

snapshot_Freq=25
sample_Freq=0
val_Freq=1


n_sentence=100000
n_batch=128 
n_maxlen=30 ##max length of sentences in tokenizing
n_gen_maxlen=200 ## max length of generated sentences
n_words=100000 ## max number of words in dictionary
dim_word=1024# dimention of word embedding 

n_u = dim_word
n_h = 1024 ## number of hidden nodes in encoder


stochastic=False
use_dropout=True
verbose=1

L1_reg=0
L2_reg=0

print 'Loading data...'

load_file='data/imdb_sen_count.pkl'

train, valid, test = load_data(load_file,n_words=n_words, valid_portion=0.05,
                               maxlen=n_maxlen,max_lable=None)
'''
train_x=train[0]     
train_y=train[1] 

train_x=train_x[:10000]
train_y=train_y[:10000]

train=(train_x,train_y)
'''                          
                               
n_y = np.max((np.max(train[1]),np.max(valid[1]))) + 1

print 'number of classes: %i'%n_y
print 'number of training data: %i'%len(train[0])
print 'number of validation data: %i'%len(valid[0])

####build model
print 'Initializing model...'

mode='tr'

model = RNN(n_epochs=n_epochs,n_batch=n_batch,snapshot=snapshot_Freq,
            sample_Freq=sample_Freq,val_Freq=val_Freq,L1_reg=L1_reg,L2_reg=L2_reg)
model.add(Embedding(n_words,dim_word))            
model.add(Drop_out(0.25))
model.add(BiDirectionGRU(n_u,n_h,return_seq=False))
model.add(Drop_out())
model.add(FC_layer(n_h,n_y))
model.add(Activation('softmax'))
model.compile(optimizer=optimizer,loss=loss)



filepath='save/review3.pkl'

if mode=='tr':
    if os.path.isfile(filepath): model.load(filepath)
    print '<training data>'    
    seq,seq_mask,targets=prepare_full_data(train[0],train[1],n_maxlen)
    print '<validation data>'
    val,val_mask,val_targets=prepare_full_data(valid[0],valid[1],n_maxlen)

    model.train(seq,seq_mask,targets,val,val_mask,val_targets,verbose)
    model.save(filepath)
    
    ##draw error graph 
    plt.close('all')
    fig = plt.figure()
    ax3 = plt.subplot(111)   
    plt.plot(model.errors)
    plt.grid()
    ax3.set_title('Training error')    
    plt.savefig('error.png')
    
    
elif mode=='te':
    if os.path.isfile(filepath): model.load(filepath)
    else: 
        raise IOError('loading error...')

    tes,tes_mask,tes_targets=prepare_full_data(test[0],test[1],n_maxlen)
    
    tes=np.asarray(tes,'int32')
    tes_mask=np.asarray(tes_mask,'float32')
    
    n_train = tes.shape[1]
    n_train_batches = int(np.ceil(1.0 * n_train / n_batch))
    
    print 'Testing model ...'
    
    progbar=Progbar(n_train_batches)
    result=[]
    for idx in xrange(n_train_batches):
        batch_start=idx*n_batch
        batch_stop=np.minimum(n_train,(idx+1)*n_batch)
        r=model.predict_proba(tes[:,batch_start:batch_stop],tes_mask[:,batch_start:batch_stop])
        for i in range(len(r)):
            result.append(r[i])   
        progbar.update(idx+1)
      

    result=np.asarray(result)
    arg=np.argmax(result,axis=-1)

    print ('Acc: %f'%accuracy_score(tes_targets,arg))

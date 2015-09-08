import theano.tensor as T
import theano,os
import numpy as np
import matplotlib.pyplot as plt
from Layers import hidden,lstm,gru,BiDirectionLSTM,decoder,BiDirectionGRU,drop_out
from Model_RNN import RNN
from Load_data import load_data,prepare_full_data
from utils import Progbar
from sklearn.metrics import accuracy_score


#theano.config.exception_verbosity='high'

#theano.config.optimizer='None' 



n_epochs = 200
lr=0.001
momentum_switchover=5
learning_rate_decay=0.999
optimizer="Adam"

#RMSprop,SGD,Adagrad,Adadelta,Adam

snapshot_Freq=20
sample_Freq=15
val_Freq=2


n_sentence=100000
n_batch=512 
n_chapter=None ## unit of slicing corpus
n_maxlen=40 ##max length of sentences in tokenizing
n_gen_maxlen=20 ## max length of generated sentences
n_words=15000 ## max numbers of words in dictionary
dim_word=500  ## dimention of word embedding 

n_u = dim_word
n_h = 800 ## number of hidden nodes in encoder


stochastic=False
use_dropout=True
verbose=1

L1_reg=0
L2_reg=0.0001

print 'Loading data...'

load_file='data/imdb_sen_count.pkl'

train, valid, test = load_data(load_file,n_words=n_words, valid_portion=0.004,
                               maxlen=None)
n_y = np.max((np.max(train[1]),np.max(valid[1]),np.max(test[1]))) + 1


####build model
print 'Initializing model...'

mode='tr'

model = RNN(n_u,n_h,n_y,n_epochs,n_chapter,n_batch,n_gen_maxlen,n_words,dim_word,
            momentum_switchover,lr,learning_rate_decay,snapshot_Freq,sample_Freq,val_Freq,
            use_dropout,L1_reg,L2_reg)
            
model.add(drop_out(use_dropout,0.25))
model.add(BiDirectionGRU(n_u,n_h))
model.add(drop_out(use_dropout))
model.add(gru(n_h,n_h))
model.add(drop_out(use_dropout))
model.add(gru(n_h,n_h))
model.add(drop_out(use_dropout))
model.add(gru(n_h,n_h))
model.build()



filepath='save/review3.pkl'

if mode=='tr':
    if os.path.isfile(filepath): model.load(filepath)
        
    seq,seq_mask,targets=prepare_full_data(train[0],train[1],n_maxlen)

    val,val_mask,val_targets=prepare_full_data(valid[0],valid[1],n_maxlen)
    
    
    model.train(seq,seq_mask,targets,val,val_mask,val_targets,verbose,optimizer)
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

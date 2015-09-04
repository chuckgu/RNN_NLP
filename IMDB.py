import theano.tensor as T
import theano,os
import numpy as np
import matplotlib.pyplot as plt
from Layers import hidden,lstm,gru,BiDirectionLSTM,decoder,BiDirectionGRU
from Models_RNN import RNN
from preprocess import load_data,prepare_full_data




#theano.config.exception_verbosity='high'

#theano.config.optimizer='None' 

def sampling(i,model,input,output,seq,seq_mask,targets,stochastic,n_gen_maxlen,n_words):
    test=seq[:,i]
    test_mask=seq_mask[:,i]
    
    truth=targets[:,i]
    
    guess = model.gen_sample(test,test_mask,stochastic)
    
    print 'Input: ',' '.join(input.sequences_to_text(test))
    
    print 'Truth: ',' '.join(output.sequences_to_text(truth))
    
    prob=np.asarray(guess[0],dtype=np.float)
    
    estimate=guess[1]
    
    print 'Sample: ',' '.join(output.sequences_to_text(estimate))
    
    return prob,estimate

n_epochs = 200
lr=0.001
momentum_switchover=5
learning_rate_decay=0.999
optimizer="Adam"

#RMSprop,SGD,Adagrad,Adadelta,Adam

snapshot_Freq=50
sample_Freq=20
val_Freq=30


n_sentence=100000
n_batch=128 
n_chapter=None ## unit of slicing corpus
n_maxlen=200 ##max length of sentences in tokenizing
n_gen_maxlen=20 ## max length of generated sentences
n_words=15000 ## max numbers of words in dictionary
dim_word=1000  ## dimention of word embedding 

n_u = dim_word
n_h = 2000 ## number of hidden nodes in encoder


stochastic=False
use_dropout=True
verbose=1

print 'Loading data...'

load_file='data/imdb.pkl'

train, valid, test = load_data(load_file,n_words=n_words, valid_portion=0.02,
                               maxlen=None)
n_y = np.max(train[1]) + 1

seq,seq_mask,targets=prepare_full_data(train[0],train[1],n_maxlen)

val,val_mask,val_targets=prepare_full_data(valid[0],valid[1],n_maxlen)

####build model
print 'Initializing model...'

mode='tr'

model = RNN(n_u,n_h,n_y,n_epochs,n_chapter,n_batch,n_gen_maxlen,n_words,dim_word,
            momentum_switchover,lr,learning_rate_decay,snapshot_Freq,sample_Freq,val_Freq)
model.add(BiDirectionGRU(n_u,n_h))
model.build()



filepath='save/review.pkl'

if mode=='tr':
    if os.path.isfile(filepath): model.load(filepath)
    
    
    model.train(seq,seq_mask,targets,val,val_mask,val_targets,verbose,optimizer,use_dropout)
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

    i=20
    for j in range(i):
        k=np.random.randint(1,n_sentence)
        a=j+1
        print('\nsample %i >>>>'%a)
        prob,estimate=sampling(k,model,input,output,seq,seq_mask,targets,stochastic,n_gen_maxlen,n_words)
 

 



    

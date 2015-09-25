import theano.tensor as T
import theano,os
import numpy as np
import matplotlib.pyplot as plt
from Layers import hidden,lstm,gru,BiDirectionLSTM,decoder,BiDirectionGRU,drop_out,Embedding,FC_layer,Pool
from Model_NMT import ENC_DEC
from Load_data import prepare_data,load_data,load_dict

#theano.config.exception_verbosity='high'

#theano.config.optimizer='None' 


n_epochs = 50
lr=0.001
momentum_switchover=5
learning_rate_decay=0.999
optimizer="Adam" #RMSprop,SGD,Adagrad,Adadelta,Adam

snapshot_Freq=20
sample_Freq=5
val_Freq=50

n_sentence=9000
n_batch=128 
n_chapter=None ## unit of slicing corpus
n_maxlen=20 ##max length of sentences in tokenizing
n_gen_maxlen=20 ## max length of generated sentences
n_words_x=10000 ## max numbers of words in dictionary
n_words_y=10000 ## max numbers of words in dictionary
dim_word=1000  ## dimention of word embedding 

n_u = dim_word
n_h = 1000 ## number of hidden nodes in encoder

n_d = 1000 ## number of hidden nodes in decoder
n_y = dim_word

stochastic=False
use_dropout=True
verbose=1

####Load data

print 'Loading data...'

load_file='data/subscript.pkl'
dic_file='data/subscript.dict.pkl'

train, valid, test = load_data(load_file,n_words=n_words_x, valid_portion=0.00,
                               maxlen=n_maxlen)

print 'number of training data: %i'%len(train[0])
print 'number of validation data: %i'%len(valid[0])

#print '<training data>' 
seq,seq_mask,targets,targets_mask=prepare_data(train[0],train[1],n_maxlen)

targets[:,:-1]=targets[:,1:]

targets_mask[:,:-1]=targets_mask[:,1:]

worddict = dict()

worddict = load_dict(dic_file)

####build model

print 'Initializing model...'

mode='tr'

model = ENC_DEC(n_u,n_h,n_d,n_y,n_epochs,n_chapter,n_batch,n_gen_maxlen,n_words_x,n_words_y,dim_word,
            momentum_switchover,lr,learning_rate_decay,snapshot_Freq,sample_Freq,val_Freq,use_dropout)
model.add(BiDirectionGRU(n_u,n_h))
model.add(decoder(n_h,n_d,n_y))
model.build()



filepath='save/sub.pkl'

if mode=='tr':
    if os.path.isfile(filepath): model.load(filepath)
    
    
    model.train(seq,seq_mask,targets,targets_mask,worddict,verbose,optimizer)
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
        
'''
    i=20
    for j in range(i):
        k=np.random.randint(1,n_sentence)
        a=j+1
        print('\nsample %i >>>>'%a)
        prob,estimate=sampling(k,model,input,output,seq,seq_mask,targets,stochastic,n_gen_maxlen,n_words)
 


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
'''


    
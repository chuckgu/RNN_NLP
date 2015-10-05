import numpy as np
from lib.Layers import Drop_out,Embedding,FC_layer,Pool,Activation,Flatten
from lib.Recurrent_Layers import Hidden,LSTM,GRU,BiDirectionLSTM,Decoder,BiDirectionGRU
from lib.Model import NN_Model
from external.mnist import load_data


np.random.seed(1337)  # for reproducibility

from lib.Convolutional_Layer import Convolution2D, MaxPooling2D
import external.np_utils

'''
    Train a simple convnet on the MNIST dataset.

    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

    Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
'''

batch_size = 128
nb_classes = 10
nb_epoch = 12

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 28, 28
# number of convolutional filters to use
nb_filters = 32
# level of pooling to perform (POOL x POOL)
nb_pool = 2
# level of convolution to perform (CONV x CONV)
nb_conv = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.reshape(X_train.shape[0], 1, shapex, shapey)
X_test = X_test.reshape(X_test.shape[0], 1, shapex, shapey)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = NN_Model(n_epochs=nb_epoch,n_batch=batch_size,val_Freq=1)

model.add(Convolution2D(nb_filters, 1, nb_conv, nb_conv, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(nb_pool, nb_pool)))
model.add(Drop_out(0.25))

model.add(Flatten())
# the resulting image after conv and pooling is the original shape
# divided by the pooling with a number of filters for each "pixel"
# (the number of filters is determined by the last Conv2D)
model.add(FC_layer(nb_filters * (shapex / nb_pool) * (shapey / nb_pool), 128))
model.add(Activation('relu'))
model.add(Drop_out(0.5))

model.add(FC_layer(128, nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adadelta',mask=False)

model.train(X_train, None , Y_train, X_test, None, Y_test)
#score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
##print('Test score:', score[0])
#print('Test accuracy:', score[1])


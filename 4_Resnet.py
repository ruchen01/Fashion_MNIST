import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import he_normal
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from keras import optimizers
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


'''The code is adapted from original ResNet50, but uses less number of filters and 40 layers since the fashion MNIST image size is much smaller and is of gray scale. The test accuracy reaches 90% very easily. However, fine tuning of hyperparameters is needed for better  performance. It seems it suffers overfitting currently, judging from the relatively large gap of test and train accuracy. ''' 


# put the original *.gz files under 'your ipython notework location/data/fashion/'
# Otherwise it'll download MNIST instead!
fashion_mnist = input_data.read_data_sets('data/fashion', one_hot = 'True')



def input(X_orig, Y_orig, m):
    X = X_orig[0:m]
    X /= 255
    X = X.reshape(m, 28, 28, 1)
    Y = Y_orig[0:m]
    return X, Y



m_train_all = len(fashion_mnist.train.labels)
m_test_all = len(fashion_mnist.test.labels)
print('total number of training example (not all are used)', m_train_all)
print('total number of test example (not all are used)', m_test_all)
X_train, Y_train = input(fashion_mnist.train.images, fashion_mnist.train.labels, m=55000)
print('train size', X_train.shape, Y_train.shape)
X_test, Y_test = input(fashion_mnist.test.images, fashion_mnist.test.labels, m=10000)
print('test size', X_test.shape, Y_test.shape)



def identity_block(X, f, filters, stage, block):
 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = he_normal(seed=None))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
        
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = he_normal(seed=None))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = he_normal(seed=None))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X = Add()([X , X_shortcut])
    X = Activation('relu')(X)
        
    return X



def convolutional_block(X, f, filters, stage, block, s = 2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',padding = 'valid', kernel_initializer = he_normal(seed=None))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path 
    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b', padding = 'same',kernel_initializer = he_normal(seed=None))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path 
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', padding = 'valid',kernel_initializer = he_normal(seed=None))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', padding = 'valid', kernel_initializer = he_normal(seed=None))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    X = Add()([X , X_shortcut])
    X = Activation('relu')(X)
        
    return X



def ResNet(input_shape = (28, 28, 1), classes = 10):
    
    X_input = Input(input_shape)

    X = ZeroPadding2D((2, 2))(X_input)
    
    # Stage 1
    X = Conv2D(32, (5, 5), strides = (1, 1), name = 'conv1', kernel_initializer = he_normal(seed=None))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    print('1, X',X.shape)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [32, 32, 64], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [32, 32, 64], stage=2, block='b')
    X = identity_block(X, 3, [32, 32, 64], stage=2, block='c')
    print('2, X',X.shape)

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [32, 32, 128], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [32, 32, 128], stage=3, block='b')
    X = identity_block(X, 3, [32, 32, 128], stage=3, block='c')
    X = identity_block(X, 3, [32, 32, 128], stage=3, block='d')
    X = identity_block(X, 3, [32, 32, 128], stage=3, block='e')
    X = identity_block(X, 3, [32, 32, 128], stage=3, block='f')
    print('3, X',X.shape)

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='c')
    print('4, X',X.shape)

    # AVGPOOL 
    X = AveragePooling2D((2,2), name = 'avg_pool')(X)
    print('average pool, X',X.shape)    
    
    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = he_normal(seed=None))(X)   
    
    model = Model(inputs = X_input, outputs = X, name='ResNet')

    return model



model = ResNet(input_shape = (28, 28, 1), classes = 10)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

#model = load_model('my_model.h5')

model.fit(X_train, Y_train, epochs = 2, batch_size = 64)
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

model.save('my_model.h5')






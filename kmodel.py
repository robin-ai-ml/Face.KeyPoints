from __future__ import division

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def create_model(useDropout=False):
    '''
    网络的输入为96x96的单通道灰阶图像, 输出30个值, 代表的15个关键点的横坐标和纵坐标
    '''
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), input_shape=(96,96,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if useDropout:
        model.add(Dropout(0.1))
        
    model.add(Convolution2D(64,(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    if useDropout:
        model.add(Dropout(0.1))
    
    model.add(Convolution2D(128,(2,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if useDropout:
        model.add(Dropout(0.1))
        
    model.add(Flatten())
    
    model.add(Dense(500, activation='relu'))

    if useDropout:
        model.add(Dropout(0.1))
        
    model.add(Dense(500, activation='relu'))
    if useDropout:
        model.add(Dropout(0.1))
        
    model.add(Dense(30))

    return model


def compile_model(model):
    sgd = SGD(lr=0.01,momentum = 0.9,nesterov=True)
    optimizer = sgd
    loss = "mean_squared_error"
    metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

'''
def train_model(model, X_train, y_train):
   return model.fit(X_train, y_train, epochs=100, batch_size=200, verbose=1, validation_split=0.2)
'''

def train_model(model, modifier, train, validation,
        batch_size=32,epochs=2000,print_every=10,patience=np.Inf):
    '''
    model :        keras model object
    Modifier:      DataModifier() object
    train:         tuple containing two numpy arrays (X_train,y_train)
    validation:    tuple containing two numpy arrays (X_val,y_val)
    patience:      The back propagation algorithm will stop if the val_loss does not decrease 
                   after  epochs
    '''
    
    ## manually write fit method
    X_train,y_train = train
    X_val, y_val    = validation
    
    generator = ImageDataGenerator()
    
    history = {"loss":[],"val_loss":[]}
    for e in range(epochs):
        if e % print_every == 0:
            print('Epoch {:4}:'.format(e)), 
        ## -------- ##
        ## training
        ## -------- ##
        batches = 0
        loss_epoch = []
        for X_batch, y_batch in generator.flow(X_train, y_train, batch_size=batch_size):
            X_batch, y_batch = modifier.fit(X_batch, y_batch)
            hist = model.fit(X_batch, y_batch,verbose=False,epochs=1)
            loss_epoch.extend(hist.history["loss"])
            batches += 1
            if batches >= len(X_train) / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break   
        loss = np.mean(loss_epoch)
        history["loss"].append(loss)
        ## --------- ##
        ## validation
        ## --------- ##
        y_pred = model.predict(X_val)
        val_loss = np.mean((y_pred - y_val)**2)
        history["val_loss"].append(val_loss)
        if e % print_every == 0:
            print("loss - {:6.5f}, val_loss - {:6.5f}".format(loss,val_loss))
        min_val_loss = np.min(history["val_loss"])
        ## Early stopping
        if patience is not np.Inf:
            if np.all(min_val_loss < np.array(history["val_loss"])[-patience:]):
                break
    return(history)



def save_model(model, fileName):
    model.save(fileName + '.h5')

def load_trained_model(fileName):
    return load_model(fileName + '.h5')

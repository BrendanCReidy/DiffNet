from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow as tf
import BatchGenerator as gen
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
import random
from keras.layers.core import Lambda
import matplotlib.pyplot as plt
from keras import backend as K
from keras import regularizers
import sys, os
import tensorflow.keras.backend as kb
from sklearn.utils import shuffle
os.environ["CUDA_VISIBLE_DEVICES"]="0"

num_classes = 2
num_permutations = 3
x_shape = np.zeros((32*(num_permutations+1),32,3)).shape
weight_decay = 0.0005

def custom_loss(y_actual,y_pred):
    custom_loss=kb.square(y_actual-y_pred)
    return custom_loss

def build_model():
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                    input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('sigmoid'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def sameClass(arr1, arr2):
    return not (False in (arr1==arr2))


val_acc_metric=tf.keras.metrics.Accuracy()

def step(X, y):
    # keep track of our gradients
    with tf.GradientTape() as tape:
    # make a prediction using the model and then calculate the
    # loss
        pred = model(X)
        loss = categorical_crossentropy(y, pred)
    # calculate the gradients using our tape and then update the
    # model weights
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

def getAccuracy(X, y):
    # keep track of our gradients
    pred = model.predict(X)
    residuals = np.argmax(pred,1)!=np.argmax(y,1)
    acc = sum(residuals)/len(residuals)
    return acc

bar_width = 50
def initProgress():
    sys.stdout.write("[%s]" % (" " * bar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (bar_width+1))



def displayProgress(current,maxSize):
    current_size = int((current / maxSize)*bar_width)
    for i in range(current_size):
        sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("\b" * (current_size+1))
    sys.stdout.flush()

def endProgress():
    for i in range(bar_width):
        sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.write("\b" * (bar_width+1))
    sys.stdout.flush()
    
        



EPOCHS = 50
BS = 64
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20
# load the MNIST dataset
print("[INFO] loading CIFAR dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/((float) (255))
x_test = x_test/((float) (255))

keys = np.argsort(y_train, axis=0, kind="mergesort")
new_y_train = np.zeros((y_train.shape[0],1))
new_x_train = np.zeros((x_train.shape[0],32,32,3))
for i in range(0, y_train.shape[0]):
    new_y_train[i][0] = y_train[keys[i]][0]
    new_x_train[i] = x_train[keys[i]]

x_train = new_x_train
y_train = new_y_train

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

batchGen = gen.BatchGenerator(x_test, y_test,10,num_permutations,10)
batchGen.start()

#"""
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

x_test,y_test = shuffle(x_test, y_test, random_state=0)

x_test = x_test[0:1000]
y_test = y_test[0:1000]

print("[INFO] creating test dataset...")
#x_test,y_test,_ = getBatch(x_test,y_test,0,1000)
#np.save("x_test", x_test)
#np.save("y_test", y_test)

print("[INFO] creating model...")
model = build_model()
opt = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compute the number of batch updates per epoch
# loop over the number of epochs
offset = 0
size = 10
batch_size = 100

for epoch in range(0, EPOCHS):
    # show the current epoch number
    print("[INFO] starting epoch {}/{}...".format(epoch + 1, EPOCHS), end="")
    #x_batch,y_batch,offset = getBatch(x_train,y_train,offset,10)
    #x_batch,y_batch = shuffle(x_batch, y_batch, random_state=0)
    print()
    initProgress()
    wT = 0
    while batchGen.isEmpty():
        #print("Waiting...")
        wT+=1
    x_batch,y_batch = batchGen.getNext()
    for i in range(0, batch_size):
        #rint("Getting batches")
        #print("Training")
        step(x_batch, y_batch)
        while batchGen.isEmpty():
            #print("Waiting...")
            wT+=1
        x_batch,y_batch = batchGen.getNext()
        #print("Shuffling")
        x_batch,y_batch = shuffle(x_batch, y_batch, random_state=0)
        displayProgress(i,batch_size)
    val_acc = getAccuracy(x_test, y_test)
    acc = getAccuracy(x_batch, y_batch)
    endProgress()
    print("Acc: ", acc, "\tVal Acc: ", val_acc)
#"""
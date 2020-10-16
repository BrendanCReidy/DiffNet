import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10

def showImage(img):
    f, axarr = plt.subplots(1, 1)
    f.set_size_inches(16, 6)
    axarr.imshow(img)
    plt.show()


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

wT = 0
while batchGen.isEmpty():
    #print("Waiting...")
    wT+=1

x_batch,y_batch = batchGen.getNext()
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import random
from tempfile import TemporaryFile
from PIL import Image
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/((float) (255))
x_test = x_test/((float) (255))

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


def numpy2pil(np_array: np.ndarray) -> Image:
    """
    Convert an HxWx3 numpy array into an RGB Image
    """

    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img

def sameSize(arr1, arr2):
    if not len(arr1.shape)==len(arr2.shape):
        return False
    for i in range(len(arr1.shape)):
        if not arr1.shape[i]==arr2.shape[i]:
            return False

    return True


def combine(arr1, arr2):
    if not sameSize(arr1,arr2):
        print("Arrays are not the same size!")
        return None
    x_size = arr1.shape[0]
    y_size = arr1.shape[1]
    z_size = arr1.shape[2]

    arr = np.zeros((2*x_size,y_size,z_size))
    arr.astype('float32')
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                arr[x][y][z] = arr1[x][y][z]
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                arr[x_size + x][y][z] = arr2[x][y][z]

    return arr


def sameClass(arr1, arr2):
    return not (False in (arr1==arr2))



def generateForClass(x_data, y_data, current_class, maxPerClass):

    num_classes = len(y_data[0])

    total = (num_classes-1)*maxPerClass*2

    x_combined = np.zeros((total,64,32,3))
    y_combined = np.zeros((total,2))
    z_combined = np.zeros((total,2,10))
    left_index = 0
    used_right = []
    k = 0
    for _ in range(maxPerClass):
        while not sameClass(current_class,y_data[left_index]):
            left_index+=1
            if(left_index>=len(y_data)):
                print("Out of range!")
                return None
        current_left = x_data[left_index]
        used_right_same = []
        for ind in range(num_classes-1):
            right_index = getRight(x_data, y_data, current_class, left_index, used_right_same)
            used_right_same.append(right_index)
            x_combined[k] = combine(x_data[left_index], x_data[right_index])
            y_combined[k][0] = 1
            y_combined[k][1] = 0
            z_combined[k][0] = current_class
            z_combined[k][1] = current_class
            k+=1
        for ind in range(num_classes):
            new_class = np.zeros(10)
            new_class[ind] = 1
            if(sameClass(current_class,new_class)):
                continue
            right_index = getRight(x_data, y_data, new_class, left_index, used_right)
            used_right.append(right_index)
            x_combined[k] = combine(x_data[left_index], x_data[right_index])
            y_combined[k][0] = 0
            y_combined[k][1] = 1
            print(y_combined[k])
            z_combined[k][0] = current_class
            z_combined[k][1] = new_class
            k+=1
        left_index+=1
    return x_combined, y_combined, z_combined

def generateForAllClasses(x_data, y_data, permutations):

    x_ret = np.zeros((0,64,32,3))
    y_ret = np.zeros((0,2))
    z_ret = np.zeros((0,2,10))
    num_classes = len(y_data[0])
    total = (num_classes-1)*permutations*2*num_classes
    print("Creating: " + str(total))
    for ind in range(num_classes):
        new_class = np.zeros(10)
        new_class[ind] = 1
        x_new, y_new, z_new = generateForClass(x_data, y_data, new_class, permutations)
        x_ret = np.concatenate((x_ret, x_new), axis=0)
        y_ret = np.concatenate((y_ret, y_new), axis=0)
        z_ret = np.concatenate((z_ret, z_new), axis=0)
        print(x_ret.shape)
    return x_ret, y_ret, z_ret

def getRight(x_data, y_data, right_class, current_index, used_right):
    num = random.randint(0, len(y_data)-1)
    while (not sameClass(right_class,y_data[num])) or (num==current_index) or (num in used_right):
        num = random.randint(0, len(y_data)-1)
    return num


f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)

"""
img = combine(x_test[0], x_test[1])
print(img.shape)

for i in range(5):
    #img = x_train[i]
    print(img.shape)
    axarr[i].imshow(img)
plt.show()
"""
new_x_test,new_y_test,new_z_test = generateForAllClasses(x_test, y_test, 100)
np.save("test_x2", new_x_test)
np.save("test_y2", new_y_test)
np.save("test_z2", new_x_test)
"""
#"""

new_x_train,new_y_train,new_z_train = generateForAllClasses(x_train, y_train, 100)
np.save("train_x2", new_x_train)
np.save("train_y2", new_y_train)
np.save("train_z2", new_z_train)
#"""
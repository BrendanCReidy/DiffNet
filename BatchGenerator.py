import numpy as np
import random
import threading
import time
import ray
import os

class BatchGenerator(threading.Thread):
    def __init__(self,x_data,y_data,batchSize=10,permutations=3,numThreads=1):
        threading.Thread.__init__(self)
        ray.init()
        self.numThreads = numThreads
        self.batchSize = batchSize
        self.offset = 0
        self.x_data = x_data
        self.y_data = y_data
        self.permutations = permutations
        self.batches = []
        self.threads = []

    def run(self):
        self.threads = []
        for i in range(self.numThreads):
            newThread = DataGen(self.x_data,self.y_data,self.batchSize,self.permutations,self.offset)
            self.threads.append(newThread)
            self.offset+=self.batchSize
        self.schedule()

    def schedule(self):
        while True:
            tIDs = []
            for thread in self.threads:
                # begin thread
                tID = thread.schedule.remote(thread)
                tIDs.append(tID)

            batches = ray.get(tIDs)
            for batch in batches:
                self.batches.append(batch)

    def isEmpty(self):
        return (len(self.batches)==0)

    def getNext(self):
        if(len(self.batches)==0):
            return None
        return self.batches.pop(0)

class DataGen():
    def __init__(self,x_data,y_data,batchSize=10,permutations=3,offset=0):
        self.batchSize = batchSize
        self.offset = offset
        self.x_data = x_data
        self.y_data = y_data
        self.permutations = permutations
        self.batches = []

    def isEmpty(self):
        return (len(self.batches)==0)

    def getNext(self):
        if(len(self.batches)==0):
            return None
        return self.batches.pop(0)

    @ray.remote
    def schedule(self):
        x_batch,y_batch = self.getBatch()
        self.offset+=self.batchSize
        return (x_batch,y_batch)

    def getBatch(self):
        x_data = self.x_data
        y_data = self.y_data
        size = self.batchSize
        offset = self.offset
        num_classes = len(y_data[0])
        intervals = len(x_data) / num_classes
        num_batches = num_classes*2*size
        num_permutations = self.permutations

        x_batch = np.zeros((num_batches,128,32,3))
        y_batch = np.zeros((num_batches,2))

        k=0
        for j in range(size):
            if(offset >= intervals):
                offset=0
            for i in range(num_classes):
                index = (int) (i*intervals + offset)
                img = x_data[index]
                start = i*intervals
                end = (i+1)*intervals
                blocked = [index]
                for _ in range(num_permutations) :
                    r = getRandom(start, end-1, blocked)
                    blocked.append(r)
                    img = combine(img, x_data[r])
                x_batch[k] = img
                y_batch[k] = np.array([1,0])
                k+=1
                img = x_data[index]
                blocked = [i]
                otherClass = getRandom(0,num_classes-1,blocked)
                start = otherClass*intervals
                end = (otherClass+1)*intervals
                blocked = [index]
                for _ in range(num_permutations) :
                    r = getRandom(start, end-1, blocked)
                    blocked.append(r)
                    img = combine(img, x_data[r])
                x_batch[k] = img
                y_batch[k] = np.array([0,1])
                k+=1
            offset+=1
        return x_batch, y_batch

def getRandom(start, end, blocked):
    r = random.randint(start, end)
    while r in blocked:
        r = random.randint(start, end)
    return r

def combine(arr1, arr2):
    x_size = arr1.shape[0]
    x_size2 = arr2.shape[0]
    y_size = arr1.shape[1]
    z_size = arr1.shape[2]

    arr = np.zeros((x_size+x_size2,y_size,z_size))
    arr.astype('float32')
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                arr[x][y][z] = arr1[x][y][z]
    for x in range(x_size2):
        for y in range(y_size):
            for z in range(z_size):
                arr[x_size + x][y][z] = arr2[x][y][z]
    return arr

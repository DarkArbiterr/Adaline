from cgi import test
import enum
from tkinter import *
from turtle import clear
import AdalineModel
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt

adalines = [AdalineModel.Adaline(784), AdalineModel.Adaline(784), AdalineModel.Adaline(784), AdalineModel.Adaline(784), AdalineModel.Adaline(784), AdalineModel.Adaline(784), AdalineModel.Adaline(784), AdalineModel.Adaline(784), AdalineModel.Adaline(784), AdalineModel.Adaline(784)]

def main():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    xTrain = [np.ndarray.flatten(t) for t in x_train]
    yTrain = np.array(y_train)
    xTest = [np.ndarray.flatten(t) for t in x_test]
    yTest = np.array(y_test)
    dataX = np.empty([60000,784])
    newDataY = np.empty([60000])
    testDataX = np.empty([10000,784])
    
    for i in range(10000):
        for j in range(784):
            if xTest[i][j] > 128:
                testDataX[i][j] = 1
            else:
                testDataX[i][j] = 0
    
    for i in range(60000):
        for j in range(784):
            if xTrain[i][j] > 128:
                dataX[i][j] = 1
            else:
                dataX[i][j] = 0
                
    for i in range(10):
        correct = 0
        dataY = np.array(yTrain)
        for j in range(60000):
            if dataY[j] == i:
                newDataY[j] = 1
            else:
                newDataY[j] = 0
        adalines[i].Learn(dataX, newDataY, i)
    
    correct = 0    
    
    for i in range(60000):
        result = []
        for j in range(10):
            result.append(adalines[j].Output(dataX[i]))
        for k in range(10):
            if result[k] == max(result):
                if k == yTrain[i]:
                    correct += 1
    print("Train Dataset: ", correct, " / 60000")
    
    correctNumbers = 0
    
    for i in range(10000):
        
        result = []
        
        for j in range(10):
            result.append(adalines[j].Output(testDataX[i].flatten()))
        for k in range(10):
            if result[k] == max(result):
                if k == yTest[i]:
                    correctNumbers += 1
        
    print("Test Dataset: ", correctNumbers, " / 10000")
            
    for i in range(10):
        errors = adalines[i].errorsList
        plt.plot(range(len(errors)), errors)
        plt.show()

if __name__ == "__main__":
    main()

import random
import numpy as np

class Adaline(object):
    def __init__(self, nrOfInputs, bias=True, iterations=15, learningEta = 0.01):
        self.nrOfInputs = nrOfInputs
        self.bias = bias
        self.errorsList = []
        self.iterations = iterations
        self.learningEta = learningEta
        self.weights = np.random.random(2 * self.nrOfInputs)
        if self.bias == False:
            self.weights = (np.random.rand(2 * self.nrOfInputs) - 0.5) / 1000
        else:
            self.weights = (np.random.rand(2 * self.nrOfInputs + 1) - 0.5) / 1000
        
    def Learn(self, dataX, dataY, a):
        for i in range(self.iterations):
            err = 0
            dataArray = list(zip(dataX, dataY))
            random.shuffle(dataArray)
            for x, y in dataArray:
                output = self.Output(x)
                x = np.concatenate([x, self.FourierTransform(x)])
                if self.bias == True:
                    x = np.concatenate([x, [1]])
                self.weights += self.learningEta * (y - output) * x
                err = err + (y - output) ** 2
            self.errorsList.append(err)
            print('Adaline nr: ', a, ": ", i)
            
    def Output(self, data):
        newData = np.concatenate([data, self.FourierTransform(data)])
        if self.bias == True:
            newData = np.concatenate([newData, [1]])
        return self.Activate(np.dot(self.weights, newData))

    def FourierTransform(self, data):
        x = np.abs(np.fft.fft(data))
        x[0] = 0
        return x / np.amax(x)

    def Activate(self, data):
        return 1 / (1 + np.exp(-data))
        


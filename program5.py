import numpy as np 
x = np.array(([2,9],[1,5],[3,6]),dtype=float)
y = np.array(([92],[86],[89]),dtype=float)

x = x/np.amax(x,axis=0)
y = y/100
class NeuralNetwork(object):
    def __init__(self):
        self.inputsize=2
        self.outputsize=1
        self.hiddensize=3
        self.w1=np.random.rand(self.inputsize,self.hiddensize)
        self.w2=np.random.rand(self.hiddensize,self.outputsize)
    def feedforward(self,x):
        self.z=np.dot(x,self.w1)
        self.z2=self.sigmoid(self.z)
        self.z3=np.dot(self.z2,self.w2)
        output=self.sigmoid(self.z3)
        return output
    def sigmoid(self,s,deriv=False):
        if(deriv==True):
            return s*(1-s)
        return 1/(1+np.exp(-s))
    def backward(self,x,y,output):
        self.output_error=y-output
        self.output_delta=self.output_error*self.sigmoid(output,deriv=True)
        self.z2_error=self.output_delta.dot(self.w2.T)
        self.z2_delta=self.z2_error*self.sigmoid(self.z2,deriv=True)
        self.w1 +=x.T.dot(self.z2_delta)
        self.w2 +=self.z2.T.dot(self.output_delta)
    def train(self,x,y):
        output=self.feedforward(x)
        self.backward(x,y,output)
NN = NeuralNetwork()
for i in range(50000):
    if(i % 100==0):
        print("Loss:"+str(np.mean(np.square(y-NN.feedforward(x)))))
    NN.train(x,y)
print("Input:"+str(x))
print("actual output:"+str(y))
print("Predicted output:"+str(NN.feedforward(x)))
print("Loss:"+str(np.mean(np.square(y-NN.feedforward(x)))))                    



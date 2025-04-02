#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
from typing import Final
import numpy as np
import pandas as pd

learningRates: Final = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
shuffleMethods: Final = [True, False]

data = pd.read_csv('optdigits.tra', sep=',', header=None)

trainingAttributes: Final = np.array(data.values)[:,:64]
trainingTargets: Final = np.array(data.values)[:,-1] 

data = pd.read_csv('optdigits.tes', sep=',', header=None)

testAttributes: Final = np.array(data.values)[:,:64]
testTargets: Final = np.array(data.values)[:,-1] 

maxPerceptronAccuracy=0.0
maxMLPAccuracy=0.0
for learningRate in learningRates:

    for shuffleMethod in shuffleMethods:
        
        for algorithm in ['Perceptron','MLP']:

            if algorithm=='Perceptron':
              clf = Perceptron(eta0=learningRate, shuffle=shuffleMethod, max_iter=1000)
            else:
              clf = MLPClassifier(activation='logistic', learning_rate_init=learningRate, 
                    hidden_layer_sizes=(25,), shuffle=shuffleMethod, max_iter=1000)
            clf.fit(trainingAttributes, trainingTargets)

            numCorrectPredictions=0
            for (testSample, testTarget) in zip(testAttributes, testTargets):
                prediction = clf.predict([testSample])
                if prediction == testTarget:
                    numCorrectPredictions+=1
            accuracy=numCorrectPredictions/len(testAttributes)
            larger=False
            if (algorithm=='Perceptron' and accuracy>maxPerceptronAccuracy):
                larger=True
                maxPerceptronAccuracy=accuracy
            if (algorithm=='MLP' and accuracy>maxMLPAccuracy):
                larger=True
                maxMLPAccuracy=accuracy
            if larger:
                print('Highest Perceptron accuracy:',maxPerceptronAccuracy,',learning rate=',learningRate,', shuffle=',shuffleMethod)
                print('Highest Multi-Layer Perceptron accuracy:',maxMLPAccuracy,',learning rate=',learningRate,', shuffle=',shuffleMethod)
            












#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

trainingData = []
normalizedTrainingData = []
normalizedTargetClasses = []
featureValues = []

with open('weather_training.csv', 'r') as csvfile:
     reader = csv.reader(csvfile)
     for i, row in enumerate(reader):
         if i > 0: #skipping the header
            trainingData.append (row)

for row in trainingData:
    i=0
    normalizedRow = []
    for value in row:
        if (i>=len(featureValues)):
            featureValues.append([])
        if (i!=0 and value not in featureValues[i]):
            featureValues[i].append(value)
        if (i!=len(row)-1 and i!=0):
            normalizedRow.append(featureValues[i].index(value))
        elif (i!=0):
            normalizedTargetClasses.append(featureValues[i].index(value))
        i+=1
    normalizedTrainingData.append(normalizedRow)

    


clf = GaussianNB(var_smoothing=1e-9)
clf.fit(normalizedTrainingData, normalizedTargetClasses)

#Reading the test data in a csv file
testData=[]
with open('weather_test.csv', 'r') as csvfile:
     reader = csv.reader(csvfile)
     for i, row in enumerate(reader):
         if i > 0: #skipping the header
            testData.append (row)

#Printing the header os the solution
#--> add your Python code here

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for testSample in testData:
    normalTestSample=[featureValues[i].index(testSample[i]) for i in range(1,len(testSample)-1)]
    predictionProbabilities=clf.predict_proba([normalTestSample])[0]
    mostProbableIndex=-1
    greatestProbability=0.0
    for i in range(0,len(predictionProbabilities)):
        if (predictionProbabilities[i]>greatestProbability):
            mostProbableIndex=i
            greatestProbability=predictionProbabilities[i]
    if (greatestProbability>=0.75):
        print('Predicted class for ',testSample,'is',featureValues[len(featureValues)-1][mostProbableIndex],"with probability",greatestProbability)

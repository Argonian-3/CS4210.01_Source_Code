#-------------------------------------------------------------------------
# AUTHOR: Aidan Zimmerman
# FILENAME: knn.py
# SPECIFICATION: Read csv file and uses knn to compute the loo-cv error rate
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

from sklearn.neighbors import KNeighborsClassifier
import csv

database = []

with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         database.append (row)

numWrongPredictions=0
for testRow in database:
    normalizedTrainingData = []
    normalizedTargetClasses = []
    normalTestSample = []
    targetClass=[]
    for row in database:
        i=0
        normalizedRow = []
        if (row!=testRow):
            normalizedTrainingData.append([float(value) for value in row[:-1]])
            if (row[-1]=='ham'):
                normalizedTargetClasses.append(0.0)
            else:
                normalizedTargetClasses.append(1.0)
        else:
            normalTestSample=[float(value) for value in row[:-1]]
            i+=1
            if (row[-1]=='ham'):
                targetClass=0.0
            else:
                targetClass=1.0
    
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(normalizedTrainingData,normalizedTargetClasses)
    predictedClass = clf.predict([normalTestSample])[0]
    if (predictedClass!=targetClass):
        numWrongPredictions+=1
print('Error rate:',numWrongPredictions/len(database))


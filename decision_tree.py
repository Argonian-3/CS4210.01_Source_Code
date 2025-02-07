#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
dataset = [] # list of lists
normalizedTrainingData = []
normalizedTargetClasses = []

featureValues = dict()
#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dataset.append(row)
         print(dataset[i-1])
         featureValues[i-1]=list()

numSamples=len(dataset)
numClasses=len(dataset[0])

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
for row in dataset:
    i=0
    for value in row:
        if (value not in featureValues[i]):
            featureValues[i].append(value)
        i+=1

for row in dataset:
    i=0
    normalizedRow = []
    for value in row:
        if (i!=len(dataset[0])-1):
            normalizedRow.append(featureValues[i].index(value))
            i+=1
        else:
            normalizedTargetClasses.append(featureValues[i].index(value))
    normalizedTrainingData.append(normalizedRow)

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
#--> addd your Python code here
# Y =

print(normalizedTrainingData)
print(normalizedTargetClasses)
#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(normalizedTrainingData, normalizedTargetClasses)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()

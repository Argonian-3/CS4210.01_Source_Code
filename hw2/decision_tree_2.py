from sklearn import tree
import csv
datasets=['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
testSamples=[]
featureValues = []

with open('contact_lens_test.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:
                testSamples.append(row)
numTestSamples=len(testSamples)
if (numTestSamples==0):
    raise Exception
normalizedTestSamples=[]
for sample in testSamples:
    i=0
    normalizedRow=[]
    for value in sample:
        if (i>=len(featureValues)):
            featureValues.append([])
        if (value not in featureValues[i]):
            featureValues[i].append(value)
        normalizedRow.append(featureValues[i].index(value))
        i+=1
    normalizedTestSamples.append(normalizedRow)
for dataset in datasets:
    trainingSamples=[]
    normalTrainingData=[]
    normalTargetClasses=[]
    with open(dataset, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: 
                trainingSamples.append (row)
    for sample in trainingSamples:
        i=0
        normalRow=[]
        for value in sample:
            if (value not in featureValues[i]):
                featureValues[i].append(value)
            if (i==len(featureValues)-1):
                normalTargetClasses.append(featureValues[i].index(value))
            else:
                normalRow.append(featureValues[i].index(value))
            i+=1
        normalTrainingData.append(normalRow)
    
    numTrials=10
    numCorrect=0
    for _ in range(numTrials):
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(normalTrainingData,normalTargetClasses)
        predictedTargetClasses=[]
        for testSample in normalizedTestSamples:
            temp=[]
            temp.append(testSample[0:len(testSample)-1])
            predictedTargetClasses.append(clf.predict(temp))
        
        numCorrect+=sum(1 for i in range(len(predictedTargetClasses)) if predictedTargetClasses[i]==normalizedTestSamples[i][len(normalizedTestSamples[i])-1])
    print('Average accuracy for ',dataset,':',numCorrect/(numTrials*numTestSamples))
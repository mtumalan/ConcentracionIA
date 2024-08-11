# KNN

import numpy as np
import codecs
import math

from statistics import mode
from operator import itemgetter

import os

def createFiles():
    current_path = os.path.dirname(os.path.abspath(__file__))
    path2 = r'\data'
    path = current_path + path2

    if not os.path.exists(path):
        os.makedirs(path)

    print("Creating samples")

    feature1 = list(np.random.normal(0, 0.2, 20))
    feature2 = list(np.random.normal(0, 0.2, 20))
    label = list(np.random.choice(['A','B'], size = 20, p = [0.5, 0.5]))

    print("Saving samples to a file")
    with codecs.open(path + r'\samples.txt', 'w', 'utf-8') as f:
        for f1, f2, l in zip(feature1, feature2, label):
            f.write(str(f1) + ',' + str(f2) + ',' + l + '\n')
            
    print("Create test samples")
    feature1 = list(np.random.normal(0, 0.2, 20))
    feature2 = list(np.random.normal(0, 0.2, 20))
    label = list(np.random.choice(['A','B'], size = 20, p = [0.5, 0.5]))

    print("Saving test samples to a file")
    with codecs.open(path + r'\KNNTest.txt', 'w', 'utf-8') as f:
        for f1, f2, l in zip(feature1, feature2, label):
            f.write(str(f1) + ',' + str(f2) + ',' + l + '\n')

def euclidean_distance(list1, list2):
    sumList = 0
    for x,y in zip(list1, list2):
        sumList += (y-x)**2
    return math.sqrt(sumList)

def classify(testList, trainingLists, trainingLabels, k):
    distances = []
    for trainingList, label in zip(trainingLists, trainingLabels):
        # Calculate the distance between the testList and the trainingList
        value = euclidean_distance(testList, trainingList)
        distances.append((value, label))
    # Sort the distances in increasing order
    distances.sort(key=itemgetter(0))
    votelabels = []
    # Take k items with lowest distances to testList
    for x in distances[:k]:
        votelabels.append(x[1])
        
    # Return the most common class label
    return mode(votelabels)

def main():
    createFiles()
    current_path = os.path.dirname(os.path.abspath(__file__))
    path2 = r'\data'
    path = current_path + path2
    trainingLists = []
    trainingLabels = []
    testLists = []
    testLabels = []
    k = 5

    with codecs.open(path + r'\samples.txt', 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split(',')
            trainingLists.append([float(parts[0]), float(parts[1])])
            trainingLabels.append(parts[2])
    with codecs.open(path + r'\KNNTest.txt', 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split(',')
            testLists.append([float(parts[0]), float(parts[1])])
            testLabels.append(parts[2])

    correct = 0
    total = 0
    for x,y in zip(testLists, testLabels):
        total += 1
        predicted = classify(x, trainingLists, trainingLabels, k)
        if predicted == y:
            correct += 1
        print("Predicted: ", predicted, "Actual: ", y)
    
    print("Accuracy: ", correct/total)
    
if __name__ == '__main__':
    main()
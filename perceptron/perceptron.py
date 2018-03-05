import numpy as np
import time
from random import shuffle
import pdb
# pdb.set_trace()
import matplotlib.pyplot as plt
import csv

def StartPredict(w, xVecData, length):
    yi = []

    for i in range(length):
        xi = xVecData[i]
        if (Predict(w, xi) <= 0):
            yi.append('<=50K')
        else:
            yi.append('>50K')

    return yi

def CreatePredictFile(testfile, importlist):
    i = 0
    with open(testfile, 'r') as csvinput:
        with open('income.test.predicted.txt', 'w') as csvoutput:
            writer = csv.writer(csvoutput)
            for row in csvinput.readlines():
                row = row[:-2]
                row = row + ', ' +importlist[i]
                writer.writerow(row.split(','))
                i += 1
    csvinput.close()
    csvoutput.close()

def Predict(x, w):
    dot = np.dot(x, w)
    return np.sign(dot)

def StartTraining(w, xVecData, indices, answers):
    numError = 0
    for i in indices:
        xi = xVecData[i]
        yi = answers[i]
        if(yi*Predict(w, xi) <= 0):
            numError += 1
    errorPercentage = round(100 * float(numError) / float(len(xVecData)), 2)
    return errorPercentage

def Perceptron(length, trainingXVectors, indices, answers, devVec, devIndices, devAnswers, learningRateVarible, shuffleOn, numEpochs, num):
    #initialize
    w = np.zeros(length)
    count = 0
    numError = 0
    minError = 100.00
    minErrorLocation = 0.00
    c = 0
    learnRate = float(1)

    print("Starting Basic Perceptron:")
    #Run five epochs
    for epoch in range(0, numEpochs):
        j = 0
        #shuffle in each epoch
        if(shuffleOn == "on"):
            shuffle(indices)

        for i in indices:
            xi = trainingXVectors[i]
            yi = float(answers[i])

            if(yi*(Predict(w, xi)) <= 0):
                w += yi * xi * learnRate
                numError += 1
                if (learningRateVarible == "on"):
                    learnRate = float(1) / (1 + numError)
            c += 1
            j += 1

            if(num == 1000):
                count += 1
                if(count == 1000):
                    count = 0
                    errorPercentage = StartTraining(w, devVec, devIndices, devAnswers)
                    if(errorPercentage < minError):
                        w_best = w
                        minError = errorPercentage
                        minErrorLocation = round(float(epoch) + j/float(len(indices)), 2)
                           
        # errorPercentage = StartTraining(w, trainingXVectors, indices, answers)
        # print( "Epoch " + str(epoch) + " Training Data Error Percentage = " + str(errorPercentage) +"%")

        errorPercentage = StartTraining(w, devVec, devIndices, devAnswers)
        print("Epoch " + str(epoch) + " Dev Data Error Percentage = " + str(errorPercentage) + "\n")

        print("Min Error in devData = " + str(minError) + "% at: " + str(minErrorLocation) + "\n")

    return w, w_best

def NaivePerceptron(length, trainingXVectors, indices, answers, devVec, devIndices, devAnswers, learningRateVarible, shuffleOn, numEpochs, num):
    #initialize
    w = np.zeros(length)
    wa = np.zeros(length)
    count = 0
    numError = 0
    minError = 100.00
    minErrorLocation = 0.00
    c = 0
    learnRate = float(1)

    error_rates = []
    location_epoch = []

    print("Starting Naive Perceptron:")
    #Run five epochs
    for epoch in range(0, numEpochs):
        j = 0
        #shuffle in each epoch
        if(shuffleOn == "on"):
            shuffle(indices)

        for i in indices:
            xi = trainingXVectors[i]
            yi = float(answers[i])

            if(yi*(Predict(w, xi)) <= 0):
                w += yi * xi * learnRate
                numError += 1
                if (learningRateVarible == "on"):
                    learnRate = float(1) / (1 + numError)
            wa += w
            c += 1
            j += 1

            if(num == 1000):
                count += 1
                if(count == 1000):
                    count = 0
                    errorPercentage = StartTraining(wa / c, devVec, devIndices, devAnswers)
                    if(errorPercentage < minError):
                        w_best = w
                        minError = errorPercentage
                        minErrorLocation = round(float(epoch) + j/float(len(indices)), 2)
                    error_rates.append(errorPercentage)
                    location_epoch.append(round(float(epoch) + j/float(len(indices)), 2))

        # errorPercentage = StartTraining(wa / c, trainingXVectors, indices, answers)
        # print( "Epoch " + str(epoch) + " Training Data Error Percentage = " + str(errorPercentage) +"%")

        errorPercentage = StartTraining(wa / c, devVec, devIndices, devAnswers)
        print("Epoch " + str(epoch) + " Dev Data Error Percentage = " + str(errorPercentage) + "\n")
        
        print("Min Error in devData = " + str(minError) + "% at: " + str(minErrorLocation) + "\n")

    plt.figure(0)
    plt.title('Error rate over time for dev set\n Average Perceptron')
    plt.xlabel('Epoch')
    plt.ylabel('Error rate')
    plt.plot(location_epoch, error_rates)
    # plt.show()

    return wa / c, w_best

def SmartPerceptron(length, trainingXVectors, indices, answers, devVec, devIndices, devAnswers, learningRateVarible, shuffleOn, numEpochs, num):
    #initialize
    w = np.zeros(length)
    wa = np.zeros(length)
    count = 0
    numError = 0
    minError = 100.00
    minErrorLocation = 0.00
    c = 0
    learnRate = float(1)

    print("Starting Smart Perceptron:")
    #Run five epoch
    for epoch in range(0, numEpochs):
        j = 0
        #shuffle in each epoch
        if(shuffleOn == "on"):
            shuffle(indices)

        for i in indices:
            xi = trainingXVectors[i]
            yi = float(answers[i])

            if(yi*(Predict(w, xi)) <= 0):
                w += yi * xi * learnRate
                wa = c * yi * xi * learnRate
                numError += 1
                if (learningRateVarible == "on"):
                    learnRate = float(1) / (1 + numError)

            c += 1
            j += 1

            if(num == 1000):
                count += 1
                if(count == 1000):
                    count = 0
                    errorPercentage = StartTraining(w - wa / c, devVec, devIndices, devAnswers)
                    if(errorPercentage < minError):
                        w_best = w
                        minError = errorPercentage
                        minErrorLocation = round(float(epoch) + j/float(len(indices)), 2)
                           
        errorPercentage = StartTraining(w - wa / c, trainingXVectors, indices, answers)
        print( "Epoch " + str(epoch) + " Training Data Error Percentage = " + str(errorPercentage) +"%")

        # errorPercentage = StartTraining(w - wa / c, devVec, devIndices, devAnswers)
        # print("Epoch " + str(epoch) + " Dev Data Error Percentage = " + str(errorPercentage) + "\n")
        
        print("Min Error in devData = " + str(minError) + "% at: " + str(minErrorLocation) + "\n")

    return w - wa / c, w_best

def AvgMIRA(length, trainingXVectors, indices, answers, devVec, devIndices, devAnswers, learningRateVarible, shuffleOn, numEpochs, num, aggressivityThreshold):
    #initialize
    w = np.zeros(length)
    wa = np.zeros(length)
    count = 0
    numError = 0
    minError = 100.00
    minErrorLocation = 0.00
    c = 0
    learnRate = float(1)

    error_rates = []
    location_epoch = []

    print("Starting Average MIRA with Threshold: " + str(aggressivityThreshold))
    for p in aggressivityThreshold:

        #Run five epochs
        for epoch in range(0, numEpochs):
            j = 0
            #shuffle in each epoch
            if(shuffleOn == "on"):
                shuffle(indices)

            for i in indices:
                xi = trainingXVectors[i]
                yi = float(answers[i])

                if(yi * np.dot(w, xi) <= p):
                    deltaW = xi * learnRate * ((yi - np.dot(w, xi)) / xi.dot(xi))
                    w += deltaW
                    wa += c * deltaW
                    numError += 1
                    if (learningRateVarible == "on"):
                        learnRate = float(1) / (1 + numError)
                c += 1
                j += 1

                if(num == 1000):
                    count += 1
                    if(count == 1000):
                        count = 0
                        errorPercentage = StartTraining(w - wa/c, devVec, devIndices, devAnswers)
                        if(errorPercentage < minError):
                            w_best = w
                            minError = errorPercentage
                            minErrorLocation = round(float(epoch) + j/float(len(indices)), 2)
                        error_rates.append(errorPercentage)
                        location_epoch.append(round(float(epoch) + j/float(len(indices)), 2))
                               
            errorPercentage = StartTraining(w - wa/c, trainingXVectors, indices, answers)
            print( "Epoch " + str(epoch) + " Training Data Error Percentage = " + str(errorPercentage) +"%")

            # errorPercentage = StartTraining(w - wa /c, devVec, devIndices, devAnswers)
            # print("Epoch " + str(epoch) + " Dev Data Error Percentage = " + str(errorPercentage) + "\n")

            print("Min Error in devData = " + str(minError) + "% at: " + str(minErrorLocation) + "\n")

        plt.figure(1)
        plt.title('Error rate over time for dev set\n Average MIRA with Threshold: ' + str(aggressivityThreshold))
        plt.xlabel('Epoch')
        plt.ylabel('Error rate')
        plt.plot(location_epoch, error_rates)
        # plt.show()

    return w - wa/c, w_best

def MIRA(length, trainingXVectors, indices, answers, devVec, devIndices, devAnswers, learningRateVarible, shuffleOn, numEpochs, num, aggressivityThreshold):
    #initialize
    w = np.zeros(length)
    count = 0
    numError = 0
    minError = 100.00
    minErrorLocation = 0.00
    c = 0
    learnRate = float(1)

    error_rates = []
    location_epoch = []

    print("Starting MIRA with Threshold: " + str(aggressivityThreshold))
    for p in aggressivityThreshold:
        # pdb.set_trace()
        #Run five epochs
        for epoch in range(0, numEpochs):
            j = 0
            #shuffle in each epoch
            if(shuffleOn == "on"):
                shuffle(indices)

            for i in indices:
                xi = trainingXVectors[i]
                yi = float(answers[i])

                if(yi * np.dot(w, xi) <= p):
                    w += xi * learnRate * ((yi - np.dot(w, xi)) / xi.dot(xi))
                    numError += 1
                    if (learningRateVarible == "on"):
                        learnRate = float(1) / (1 + numError)
                c += 1
                j += 1

                if(num == 1000):
                    count += 1
                    if(count == 1000):
                        count = 0
                        errorPercentage = StartTraining(w, devVec, devIndices, devAnswers)
                        if(errorPercentage < minError):
                            w_best = w
                            minError = errorPercentage
                            minErrorLocation = round(float(epoch) + j/float(len(indices)), 2)
                        error_rates.append(errorPercentage)
                        location_epoch.append(round(float(epoch) + j/float(len(indices)), 2))
                               
            errorPercentage = StartTraining(w, trainingXVectors, indices, answers)
            print( "Epoch " + str(epoch) + " Training Data Error Percentage = " + str(errorPercentage) +"%")

            # errorPercentage = StartTraining(w, devVec, devIndices, devAnswers)
            # print("Epoch " + str(epoch) + " Dev Data Error Percentage = " + str(errorPercentage) + "\n")

            print("Min Error in devData = " + str(minError) + "% at: " + str(minErrorLocation) + "\n")

        plt.figure(2)
        plt.title('Error rate over time for dev set\n MIRA')
        plt.xlabel('Epoch')
        plt.ylabel('Error rate')
        plt.plot(location_epoch, error_rates)
        # plt.show()

    return w, w_best

def GetTypes(occurances):
    numTypes = []
    for valueIndex in range(0, len(occurances)-1):
        valueType = occurances[valueIndex]
        temp = []
        for i in valueType:
            temp.append(i)
        temp.sort()
        numTypes.append(temp)
    return numTypes

def CountTypes(data):
    data_temp = []
    for i in range(0,len(data[0])):
        d = {}
        for line in data:
            if line[i] not in d:
                d[line[i]] = 1
            else:
                d[line[i]] += 1
        data_temp.append(d)
    return data_temp

def GetBinarySpace(numTypes):
    binaryLength = []
    for i in range(0, len(numTypes)):
        binaryLength.append(len(numTypes[i]))
    binaryLength.append(1) #plus 1 for the bias dimension
    return binaryLength

def GetBinarySpaceBin(numTypes):
    binaryLength = []
    for i in range(0, len(numTypes)):
        if i == 0 or i == 7:
            binaryLength.append(1)
        else:
            binaryLength.append(len(numTypes[i]))
    binaryLength.append(1) #plus 1 for the bias dimension
    return binaryLength

def GetIndices(data):
    indices = list()
    for i in range(0, len(data)):
        indices.append(i)
    return indices

def DataToBinary(data, numTypes, binaryLength):
    xVecs = []
    for line in data:
        xVec = GetVector(line, numTypes, binaryLength)
        xVecs.append(xVec)
        lineCheck = DataFromVector(xVec, numTypes, binaryLength)
        for p in range(0, len(lineCheck)):
            if(line[p] != lineCheck[p]):
                raise Exception ("Error")
    return xVecs

def GetVectorBin(line, numTypes, binaryLength):
    length = sum(binaryLength)
    temp = np.zeros(length)
    pointer = 0
    for i in range(0, len(line) -1 ):
        if i == 0 or i == 7:
            temp[numTypes[i].index(line[i]) + pointer] = line[i]
        else:
            temp[numTypes[i].index(line[i])+pointer] = 1
        pointer += binaryLength[i]
    temp[-1] = 1 #bias
    return temp

def GetVector(line, numTypes, binaryLength):
    length = sum(binaryLength)
    temp = np.zeros(length)
    pointer = 0
    for i in range(0, len(line) -1 ):
        temp[numTypes[i].index(line[i]) + pointer]=1
        pointer += binaryLength[i]
    temp[-1] = 1 #bias
    return temp

def DataFromVector(xVector, numTypes, binaryLength):
    dataLine = []
    pointer = 0
    j = 0
    pointer = 0
    for i in range(0, len(xVector) - 1):
        if xVector[i] == 0:
            continue
        try:
            dataLine.append(numTypes[j][i-pointer])
        except:
            raise Exception ("Error")
        pointer += binaryLength[j]
        j += 1
    return dataLine

def GetAnswers(data):
    answers = dict()
    for i in range(0, len(data)):
        line = data[i]
        if(line[-1] == '>50K'):
            answers[i] = 1
        else :
            answers[i] = -1
    return answers

def SplitData(filename):
    r = open(filename)
    data = []
    for line in r:
        temp = []
        for x in line.split(','):
            temp.append(x.strip())
        data.append(temp)
    return data

################ Main Function ################

# "on" or "off"
shuffleOn = "off"
learningRateVarible = "off"
binFeature = "off"

# Average: [0.0, 0.1, 0.5, 0.9]
aggressivityThreshold = [0.0]

trainData = SplitData("income.train.txt")
occurances = CountTypes(trainData)
numTypes = GetTypes(occurances)
binaryLength = GetBinarySpace(numTypes)
length = sum(binaryLength)
trainAnswers = GetAnswers(trainData)
trainIndices = GetIndices(trainAnswers)
trainingVectors = DataToBinary(trainData, numTypes, binaryLength)

# zero-mean
# trainingVectors = np.array(trainingVectors)
# trainingVectors -= trainingVectors.mean(0)

devData = SplitData("income.dev.txt")
devAnswers = GetAnswers(devData)
devIndices = GetIndices(devData)
devVec = DataToBinary(devData, numTypes, binaryLength)

# zero-mean
# devVec = np.array(devVec)
# devVec -= devVec.mean(0)

testData = SplitData("income.test.txt")
testIndices = GetIndices(testData)
testVec = DataToBinary(testData, numTypes, binaryLength)

# if binFeature == "on":
#     binaryLengthBin = GetBinarySpaceBin(numTypes)
#     length = sum(binaryLengthBin)
#     trainingVectorsBin = DataToBinary(trainData, numTypes, binaryLengthBin)

startTime = time.time()
algPerc, perceptron_best = Perceptron(length, trainingVectors, trainIndices, trainAnswers, devVec, devIndices, devAnswers, learningRateVarible, shuffleOn, 5, 1000)
totalTime = time.time() - startTime;
print ("Perceptron Time: " + str(totalTime) + " sec" + '\n' )

startTime = time.time()
algNaivePerc, avgPerceptron_best = NaivePerceptron(length, trainingVectors, trainIndices, trainAnswers, devVec, devIndices, devAnswers, learningRateVarible, shuffleOn, 5, 1000)
totalTime = time.time() - startTime;
print ("Perceptron Time: " + str(totalTime) + " sec" + '\n' )

startTime = time.time()
algSmartPerc = SmartPerceptron(length, trainingVectors, trainIndices, trainAnswers, devVec, devIndices, devAnswers, learningRateVarible, shuffleOn, 5, 1000)
totalTime = time.time() - startTime;
print ("Perceptron Time: " + str(totalTime) + " sec" + '\n' )

startTime = time.time()
algMIRA, MIRA_best = MIRA(length, trainingVectors, trainIndices, trainAnswers, devVec, devIndices, devAnswers, learningRateVarible, shuffleOn, 5, 1000, aggressivityThreshold)
totalTime = time.time() - startTime;
print ("Perceptron Time: " + str(totalTime) + " sec" + '\n' )

startTime = time.time()
algAvgMIRA, algAvgMIRA_best = AvgMIRA(length, trainingVectors, trainIndices, trainAnswers, devVec, devIndices, devAnswers, learningRateVarible, shuffleOn, 5, 1000, aggressivityThreshold)
totalTime = time.time() - startTime;
print ("Perceptron Time: " + str(totalTime) + " sec" + '\n' )

# testAnswers = StartPredict(avgPerceptron_best, testVec, len(testIndices))
# CreatePredictFile('income.test.txt', testAnswers)


















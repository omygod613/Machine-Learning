#!/usr/bin/env python

from __future__ import division

import sys
import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import accuracy_score
import csv

def map_data(filename, feature2index):
    data = [] # list of (vecx, y) pairs
    dimension = len(feature2index)
    for j, line in enumerate(open(filename)):
        line = line.strip()
        features = line.split(", ")
        feat_vec = np.zeros(dimension)
        for i, fv in enumerate(features[:-1]): # last one is target
            if (i, fv) in feature2index: # ignore unobserved features
                feat_vec[feature2index[i, fv]] = 1
        feat_vec[0] = 1 # bias

        data.append((feat_vec, 1 if features[-1] == ">50K" else -1))

    return data

def train(train_data, dev_data, it=1, MIRA=False, check_freq=1000, aggressive=0.9, verbose=True):

    train_size = len(train_data)
    dimension = len(train_data[0][0])
    model = np.zeros(dimension)
    totmodel = np.zeros(dimension)
    best_err_rate = best_err_rate_avg = best_positive = best_positive_avg = 1
    t = time.time()
    error_rates = []
    error_rates_avg = [] 
    location_epoch = []

    for i in xrange(1, it+1):
        print "starting epoch", i
        for j, (vecx, y) in enumerate(train_data, 1):
            s = model.dot(vecx)
            if not MIRA: # perceptron
                if s * y <= 0:
                    model += y * vecx
            else: # MIRA
                if s * y <= aggressive:
                    model += (y - s)  / vecx.dot(vecx) * vecx
            totmodel += model # stupid!
            if j % check_freq == 0:
                dev_err_rate, positive = test(dev_data, model)
                dev_err_rate_avg, positive_avg = test(dev_data, totmodel)        
                epoch_position = i-1 + j/train_size
                if dev_err_rate < best_err_rate:
                    best_err_rate = dev_err_rate
                    best_err_pos = epoch_position #(i, j)
                    best_positive = positive
                    best_model = model
                if dev_err_rate_avg < best_err_rate_avg:
                    best_err_rate_avg = dev_err_rate_avg
                    best_err_pos_avg = epoch_position #(i, j)
                    best_positive_avg = positive_avg
                    best_avg_model = totmodel
                error_rates.append(dev_err_rate)
                error_rates_avg.append(dev_err_rate_avg)
                location_epoch.append(epoch_position)

    print "training %d epochs costs %f seconds" % (it, time.time() - t)
    print "MIRA" if MIRA else "perceptron", aggressive if MIRA else "", \
        "unavg err: {:.2%} (+:{:.1%}) at epoch {:.2f}".format(best_err_rate, 
                                                              best_positive, 
                                                              best_err_pos), \
        "avg err: {:.2%} (+:{:.1%}) at epoch {:.2f}".format(best_err_rate_avg, 
                                                            best_positive_avg, 
                                                            best_err_pos_avg)
    plt.figure()
    plt.title('Error rate over time for dev set\n MIRA = %s, aggressive = %.1f' % (MIRA, aggressive))
    plt.xlabel('Epoch')
    plt.ylabel('Error rate')
    plt.plot(location_epoch, error_rates)
    plt.plot(location_epoch, error_rates_avg, 'r')
    plt.legend(['non-average', 'average'], loc='upper left')
    return best_model

def train_Pegasos(train_data, dev_data, it=1, check_freq=1000, aggressive=0.9, verbose=True, C = 1):

    train_size = len(train_data)
    dimension = len(train_data[0][0])
    model = np.ones(dimension)
    totmodel = np.zeros(dimension)
    best_err_rate = best_err_rate_avg = best_positive = best_positive_avg = 1
    train_best_err_rate = train_best_err_rate_avg = train_best_positive = train_best_positive_avg = 1
    t = time.time()
    
    error_rates = []
    error_rates_avg = [] 
    location_epoch = []

    train_error_rates = []
    train_error_rates_avg = [] 
    train_location_epoch = []

    lambd = 2 / (train_size * C)
    s = model.dot(train_data[0][0])
    cw_array = []
    sampleNum = 0
    cw = 0
    
    # ksaai = 0

    for i in xrange(1, it+1):
        # print "starting epoch", i
        lagaranges = 0
        supportvector = 0
        for j, (vecx, y) in enumerate(train_data, 1):
            
            sampleNum += 1
            s = model.dot(vecx)
            
            leaningrate = 1 / (lambd * sampleNum)
            # leaningrate = 1 / (lambd * (j + (i-1) * train_size))
            if s * y <= 1:
                model = model - leaningrate * lambd * model + leaningrate * y * vecx
                lagarange = 1 - s * y
                supportvector += 1
                # print lagarange
                # ksaai += s * y
            else:
                model = model - leaningrate * lambd * model
                lagarange = 0
            lagaranges += lagarange
            # print lagaranges
            
            totmodel += model # stupid!
            if j % check_freq == 0:
                dev_err_rate, positive = test(dev_data, model)
                # dev_err_rate_avg, positive_avg = test(dev_data, totmodel)

                train_err_rate, train_positive = test(train_data, model)
                # train_err_rate_avg, train_positive_avg = test(train_data, totmodel)

                epoch_position = i-1 + j/train_size
                train_epoch_position = i-1 + j/train_size
                ## dev
                if dev_err_rate < best_err_rate:
                    best_err_rate = dev_err_rate
                    best_err_pos = epoch_position #(i, j)
                    best_positive = positive
                    best_model = model
                # if dev_err_rate_avg < best_err_rate_avg:
                #     best_err_rate_avg = dev_err_rate_avg
                #     best_err_pos_avg = epoch_position #(i, j)
                #     best_positive_avg = positive_avg
                #     best_avg_model = totmodel
                error_rates.append(dev_err_rate)
                # error_rates_avg.append(dev_err_rate_avg)
                location_epoch.append(epoch_position)
                ## train
                if train_err_rate < train_best_err_rate:
                    train_best_err_rate = train_err_rate
                    train_best_err_pos = train_epoch_position #(i, j)
                    train_best_positive = train_positive
                # if train_err_rate_avg < train_best_err_rate_avg:
                #     train_best_err_rate_avg = train_err_rate_avg
                #     train_best_err_pos_avg = train_epoch_position #(i, j)
                #     train_best_positive_avg = train_positive_avg
                #     train_best_avg_model = totmodel
                train_error_rates.append(train_err_rate)
                # train_error_rates_avg.append(train_err_rate_avg)
                train_location_epoch.append(train_epoch_position)


        # cw = (lambd / 2) * model.dot(model) + (1 / train_size) * lagaranges 
        cw = model.dot(model) + C * lagaranges
        cw_array.append(cw)
        
        # print "epoch {:}: objective = {:.4}, train error = {:.2%}, dev error = {:.2%}".format(i - 1, cw, train_err_rate, dev_err_rate)
    # print "supportvector: %d" % supportvector
        # print "cw_array", cw_array

    # print"training %d epochs costs %f seconds" % (it, time.time() - t)
    # print"Pegasos", \
    # "unavg err: {:.2%} (+:{:.1%}) at epoch {:.2f}".format(best_err_rate,
    #                                                       best_positive,
    #                                                       best_err_pos), \
    # "avg err: {:.2%} (+:{:.1%}) at epoch {:.2f}".format(best_err_rate_avg,
    #                                                     best_positive_avg,
    #                                                     best_err_pos_avg)
    # plt.figure()
    # plt.title('Objective function over time for dev set\n Pegasos')
    # plt.xlabel('Epoch')
    # plt.ylabel('Min_Objective')
    # plt.plot(xrange(2, it+1), cw_array[1:])

    # plt.figure()
    # plt.title('Error rate over time for dev set\n Pegasos')
    # plt.xlabel('Epoch')
    # plt.ylabel('Error rate')
    # plt.plot(location_epoch, train_error_rates)
    # plt.plot(location_epoch, error_rates, 'r')
    # plt.legend(['train_set', 'dev_set'], loc='upper right')
  
    # plt.figure()
    # plt.title('Error rate over time for dev set\n Pegasos')
    # plt.xlabel('Epoch')
    # plt.ylabel('Error rate')
    # plt.plot(location_epoch, error_rates)
    # plt.plot(location_epoch, error_rates, 'r')
    # # plt.plot(location_epoch, error_rates_avg, 'r')
    # # plt.legend(['non-average', 'average'], loc='upper right')

    return best_model

def test(data, model):
    errors = sum(model.dot(vecx) * y <= 0 for vecx, y in data)
    positives = sum(model.dot(vecx) > 0 for vecx, _ in data) # stupid!
    return errors / len(data), positives / len(data)

def CreatePredictFile(testfile, answers):
    i = 0

    with open(testfile, 'r') as csvinput:
        with open('income.test.predict.txt', 'w') as csvoutput:
            writer = csv.writer(csvoutput)
            for row in csvinput.readlines():
                row = row[:-2]
                row = row + ', ' + answers[i]
                writer.writerow(row.split(','))
                i += 1
    csvinput.close()
    csvoutput.close()

def StartPredict(w, xVecData, length):
    yi = []

    for i in range(length):
        xi = xVecData[i][0]
        if (Predict(w, xi) <= 0):
            yi.append('<=50K')
        else:
            yi.append('>50K')
    return yi

def Predict(x, w):
    dot = np.dot(x, w)
    return dot

def create_feature_map(train_file):

    column_values = defaultdict(set)
    for line in open(train_file):
        line = line.strip()
        features = line.split(", ")[:-1] # last field is target
        for i, fv in enumerate(features):
            column_values[i].add(fv)

    feature2index = {(-1, 0): 0} # bias
    for i, values in column_values.iteritems():
        for v in values:            
            feature2index[i, v] = len(feature2index)

    dimension = len(feature2index)
    # print "dimensionality: ", dimension
    return feature2index

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_file, dev_file = sys.argv[1], sys.argv[2]
    else:
        train_file, dev_file, test_file, test_predict_file = "income.train.txt.5k", "income.dev.txt", "income.test.txt", "income.test.predicted.txt"

    feature2index = create_feature_map(train_file)
    train_data = map_data(train_file, feature2index)
    dev_data = map_data(dev_file, feature2index)
    test_data = map_data(test_file, feature2index)
    test_predict_data = map_data(test_predict_file, feature2index)

    # best_model = train(train_data, dev_data, it=160, MIRA=False, check_freq=1000, verbose=False)
    # print "train_err {:.2%} (+:{:.1%})".format(*test(train_data, best_model))

    # best_model = train(train_data, dev_data, it=160, MIRA=True, check_freq=1000, verbose=False, aggressive=0)
    # print "train_err {:.2%} (+:{:.1%})".format(*test(train_data, best_model))

    # best_model = train(train_data, dev_data, it=160, MIRA=True, check_freq=1000, verbose=False, aggressive=0.9)
    # print "train_err {:.2%} (+:{:.1%})".format(*test(train_data, best_model))

    print "Pegasos Algorithm"
    print " C = 1"
    best_model = train_Pegasos(train_data, dev_data, it=100, check_freq=1000, verbose=False, C=1)
    print ("train_err {:.2%} (+:{:.1%})".format(*test(train_data, best_model)))
    print ("dev_err {:.2%} (+:{:.1%})".format(*test(dev_data, best_model)))
    print ("test_predict_err {:.2%} (+:{:.1%})".format(*test(test_predict_data, best_model)))

    # testAnswers = StartPredict(best_model, test_data, len(test_data))
    # CreatePredictFile('income.test.txt', testAnswers)




    # plt.show()




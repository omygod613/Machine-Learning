import matplotlib.pyplot as plt
from collections import defaultdict
import time
import csv

startsym, stopsym = "<s>", "</s>"
import tagger


def train(trainfile, devfile, dictionary, Average=False, MultiGrams=False, epochs=10):
    weight = defaultdict(float)
    avg_weight = defaultdict(float)
    trainset = tagger.readfile(trainfile)
    c = 0
    best_dev_err = 1
    error_rates = []
    error_rates_train = []
    location_epoch = []

    if not Average:
        print('Unaverage Structure Perceptron, MultiGrams = %s' % (MultiGrams))
    else:
        print('Average Structure Perceptron, MultiGrams = %s' % (MultiGrams))
    for epoch in range(1, epochs + 1):
        errors = 0

        for wordseq, gold_tagseq in trainset:
            c += 1
            cur_tagseq = tagger.decode(wordseq, dictionary, weight, MultiGrams)

            if cur_tagseq != gold_tagseq:
                errors += 1
                phi_total = defaultdict(float)
                wordseq = [startsym] + wordseq + [stopsym]
                gold_tagseq = [startsym] + gold_tagseq + [stopsym]
                cur_tagseq = [startsym] + cur_tagseq + [stopsym]

                for i, (word, tag_gold, tag_cur) in enumerate(zip(wordseq[1:], gold_tagseq[1:], cur_tagseq[1:]), 1):
                    if tag_gold != tag_cur:
                        phi_total[('tw', tag_gold, word)] += 1  # tag(y) -> word
                        phi_total[('tw', tag_cur, word)] -= 1  # tag(z) -> word

                        if MultiGrams:
                            phi_total[('tt_1w', tag_gold, gold_tagseq[i - 1], word)] += 1
                            phi_total[('tt_1w', tag_cur, cur_tagseq[i - 1], word)] -= 1

                    if tag_gold != tag_cur or gold_tagseq[i - 1] != cur_tagseq[i - 1]:
                        phi_total[(tag_gold, gold_tagseq[i - 1])] += 1  # phi(x, y)
                        phi_total[(tag_cur, cur_tagseq[i - 1])] -= 1  # phi(x, z)

                        if MultiGrams:
                            phi_total[(tag_gold, word, wordseq[i - 1])] += 1
                            phi_total[(tag_cur, word, wordseq[i - 1])] -= 1

                            phi_total[('tt_1w_1', tag_gold, gold_tagseq[i - 1], wordseq[i - 1])] += 1
                            phi_total[('tt_1w_1', tag_cur, cur_tagseq[i - 1], wordseq[i - 1])] -= 1

                            phi_total[(tag_gold, gold_tagseq[i - 1], word, wordseq[i - 1])] += 1
                            phi_total[(tag_cur, cur_tagseq[i - 1], word, wordseq[i - 1])] -= 1

                            if i >= 2:
                                phi_total[(tag_gold, gold_tagseq[i - 2], gold_tagseq[i - 1])] += 1
                                phi_total[(tag_cur, cur_tagseq[i - 2], cur_tagseq[i - 1])] -= 1
                            if i >= 3:
                                phi_total[(tag_gold, gold_tagseq[i - 3], gold_tagseq[i - 2], gold_tagseq[i - 1])] += 1
                                phi_total[(tag_cur, cur_tagseq[i - 3], cur_tagseq[i - 2], cur_tagseq[i - 1])] -= 1

                    if not Average:
                        for e in phi_total.keys():
                            weight[e] += phi_total[e]
                    else:
                        for e in phi_total.keys():
                            # avg_weight
                            weight[e] += phi_total[e]
                            avg_weight[e] += c * phi_total[e]
                        # update(avg_weight, phi_total, c)

        if Average:
            for e in weight:
                avg_weight[e] = weight[e] - avg_weight[e] / c

        if not Average:
            train_err = tagger.test(trainfile, dictionary, weight, MultiGrams)
            dev_err = tagger.test(devfile, dictionary, weight, MultiGrams)

            if best_dev_err > dev_err:
                best_dev_err = dev_err
                best_epoch = epoch
                best_weight = weight

            error_rates.append(dev_err)
            error_rates_train.append(train_err)
            location_epoch.append(epoch)
            print("epoch %d, updates %d, feature = %d, train_err = %.2f%%, dev_err = %.2f%%" % (
                epoch, errors, num_feature(weight), train_err * 100, dev_err * 100))

        else:
            train_avg_err = tagger.test(trainfile, dictionary, avg_weight, MultiGrams)
            dev_avg_err = tagger.test(devfile, dictionary, avg_weight, MultiGrams)

            if best_dev_err > dev_avg_err:
                best_dev_err = dev_avg_err
                best_epoch = epoch
                best_weight = weight

            error_rates.append(dev_avg_err)
            error_rates_train.append(train_avg_err)
            location_epoch.append(epoch)
            print("epoch %d, updates %d, feature = %d, train_err = %.2f%%, dev_avg_err = %.2f%%" % (
                epoch, errors, num_feature(weight), train_avg_err * 100, dev_avg_err * 100))
    if not Average:
        print("The best dev_err = %.2f%% at %d epoch" % (best_dev_err * 100, best_epoch))
    else:
        print("The best dev_avg_err = %.2f%% at %d epoch" % (best_dev_err * 100, best_epoch))

    return error_rates_train, error_rates, location_epoch, best_dev_err, best_weight


def num_feature(weight):
    num_f = 0
    for idx, value in weight.items():
        if value != 0:
            num_f += 1
    return num_f


def predict_lable(testfile, dictionary, weight, MultiGrams):
    with open(testfile, 'r') as csvinput:
        # with open('test.lower.unk.best', 'w') as csvoutput:
        with open('dev.lower.unk.best', 'w') as csvoutput:
            writer = csv.writer(csvoutput)
            for line in csvinput.readlines():
                wt = ''
                words = [x for x in line.split()]
                predict_tag = tagger.decode(words, dictionary, weight, MultiGrams)
                for word, tag in zip(words, predict_tag):
                    wt += word + '/' + tag + ' '
                # print(wt.split(','))
                writer.writerow(wt.split(','))
    csvinput.close()
    csvoutput.close()

def remove_lable(devfile):
    with open(devfile, 'r') as csvinput:
        with open('dev.txt.lower.unk.unlabeled', 'w') as csvoutput:
            writer = csv.writer(csvoutput)

            for line in csvinput.readlines():
                w = ''
                words = [x for x in line.split()]

                for word in words:
                    # print(word.split('/')[0])
                    w += word.split('/')[0] + ' '
                writer.writerow(w.split(','))
    csvinput.close()
    csvoutput.close()


if __name__ == "__main__":
    num_epochs = 10

    # trainfile, devfile, testfile = sys.argv[1:4]
    trainfile, devfile, testfile, test_predicted_file, dev_unlabeled, dev_predicted_file = "train.txt.lower.unk", "dev.txt.lower.unk", "test.txt.lower.unk.unlabeled", "test.lower.unk.best", "dev.txt.lower.unk.unlabeled", "dev.lower.unk.best"

    # remove_lable(devfile)

    dictionary, weight = tagger.mle(trainfile)

    # print("train_err {0:.2%}".format(tagger.test(trainfile, dictionary, weight)))
    # print("dev_err {0:.2%}".format(tagger.test(devfile, dictionary, weight)))

    t = time.time()
    error_rates_train, error_rates, location_epoch, best_train_err, best_weight = train(trainfile, devfile, dictionary,
                                                                                        False, False, num_epochs)
    print('Spend %.3f seconds' % (time.time() - t))

    t = time.time()
    error_rates_train_avg, error_rates_avg, location_epoch_avg, best_train_err, best_avg_weight = train(trainfile,
                                                                                                        devfile,
                                                                                                        dictionary,
                                                                                                        True, False,
                                                                                                        num_epochs)
    print('Spend %.3f seconds' % (time.time() - t))

    t = time.time()
    error_rates_train, error_rates, location_epoch, best_train_err, best_weight_mulitgram = train(trainfile, devfile, dictionary,
                                                                                        False, True, num_epochs)
    print('Spend %.3f seconds' % (time.time() - t))

    t = time.time()
    error_rates_train_avg, error_rates_avg, location_epoch_avg, best_train_err, best_avg_weight_mulitgram = train(trainfile,
                                                                                                        devfile,
                                                                                                        dictionary,
                                                                                                        True, True,
                                                                                                        num_epochs)
    print('Spend %.3f seconds' % (time.time() - t))



    # predict_lable(testfile, dictionary, best_avg_weight_mulitgram, True)
    # predict_lable(dev_unlabeled, dictionary, best_avg_weight_mulitgram, True)

    # print('Starting evaluating predicted test-best set using best model:')
    # error_rates_train, error_rates, location_epoch, best_train_err, best_weight = train(trainfile, test_predicted_file, dictionary, False, False, num_epochs)
    # error_rates_train, error_rates, location_epoch, best_train_err, best_weight = train(trainfile, test_predicted_file, dictionary, True, False, num_epochs)
    # error_rates_train_avg, error_rates_avg, location_epoch_avg, best_train_err, best_avg_weight = train(trainfile, test_predicted_file, dictionary, False, True, num_epochs)
    # error_rates_train_avg, error_rates_avg, location_epoch_avg, best_train_err, best_avg_weight = train(trainfile, test_predicted_file, dictionary, True, True, num_epochs)
    # print('\n')
    #
    # print('Starting evaluating predicted dev-best set using best model:')
    # error_rates_train, error_rates, location_epoch, best_train_err, best_weight = train(trainfile, dev_predicted_file, dictionary, False, False, num_epochs)
    # error_rates_train, error_rates, location_epoch, best_train_err, best_weight = train(trainfile, dev_predicted_file, dictionary, True, False, num_epochs)
    # error_rates_train_avg, error_rates_avg, location_epoch_avg, best_train_err, best_avg_weight = train(trainfile, dev_predicted_file, dictionary, False, True, num_epochs)
    # error_rates_train_avg, error_rates_avg, location_epoch_avg, best_train_err, best_avg_weight = train(trainfile, dev_predicted_file, dictionary, True, True, num_epochs)

    # plt.figure()
    # plt.title('Error rate over time for train and dev set\n')
    # plt.xlabel('Epoch')
    # plt.ylabel('Error rate')
    # plt.plot(location_epoch, error_rates)
    # plt.plot(location_epoch_avg, error_rates_avg)
    # plt.plot(location_epoch, error_rates_train)
    # plt.plot(location_epoch_avg, error_rates_train_avg)
    # plt.legend(['non-average: dev', 'average: dev', 'non-average: train', 'average: train'], loc='upper right')
    # plt.show()

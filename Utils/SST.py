import os
import numpy as np

class SST_Dataset(object):
    """
    Stanford Sentiment Treebank
    """

    def __init__(self, DIR, binary=False, one_hot_label=True):
        self.binary = binary
        self.one_hot_label = one_hot_label
        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtrain)

    def getxy(self, sent):
        xy = sent.split(' ||| ')
        return int(xy[1].strip()), xy[0]

    def to_array(self, integer, num_labels):
        return np.array(np.eye(num_labels)[integer])

    def remove_neutral(self, data):
        final = []
        for y, x in data:
            if y in [0, 1]:
                final.append((0, x))
            elif y in [3, 4]:
                final.append((1, x))
        return final


    def get_regression_labels(self, ytrain):
        reg_labels = []
        for y in ytrain:
            if y in [0, 1]:
                reg_labels.append((0))
            elif y in [3, 4]:
                reg_labels.append((1))
            else:
                reg_labels.append((0.5))
        return reg_labels


    def open_data(self, DIR):
        if self.binary:
            train = open(os.path.join(DIR, 'sent+phrase.binary.clean.train'))
        else:
            train = open(os.path.join(DIR, 'sent+phrase.clean.train'))

        dev = open(os.path.join(DIR, 'raw.clean.dev'))
        test = open(os.path.join(DIR, 'raw.clean.test'))

        train_data = [self.getxy(x) for x in train]
        if self.binary:
            train_data = self.remove_neutral(train_data)
        ytrain, Xtrain = zip(*train_data)
        Xtrain = [sent.split() for sent in Xtrain]


        dev_data = [self.getxy(x) for x in dev]
        if self.binary:
            dev_data = self.remove_neutral(dev_data)
        ydev, Xdev = zip(*dev_data)
        Xdev = [sent.split() for sent in Xdev]

        test_data = [self.getxy(x) for x in test]
        if self.binary:
            test_data = self.remove_neutral(test_data)
        ytest, Xtest = zip(*test_data)
        Xtest = [sent.split() for sent in Xtest]

        if self.one_hot_label is True:
            if self.binary:
                ytrain = [self.to_array(y, 2) for y in ytrain]
                ydev = [self.to_array(y, 2) for y in ydev]
                ytest = [self.to_array(y, 2) for y in ytest]
            else:
                ytrain = [self.to_array(y, 5) for y in ytrain]
                ydev = [self.to_array(y, 5) for y in ydev]
                ytest = [self.to_array(y, 5) for y in ytest]

        Xtrain = np.array(Xtrain)
        Xdev = np.array(Xdev)
        Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

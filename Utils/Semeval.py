import numpy as np
import os
from nltk.tokenize import TweetTokenizer


def rem_urls(tokens):
    final = []
    for t in tokens:
        if t.startswith('@') or t.startswith('http') or t.find('www.') > -1 or t.find('.com') > -1:
            pass
        elif t[0].isdigit():
            final.append('NUMBER')
        else:
            final.append(t)
    return final


class Semeval_Dataset():
    def __init__(self, DIR, one_hot_label=True, binary=False):
        self.one_hot_label = one_hot_label
        self.binary = binary

        Xtrain, Xdev, Xtest, ytrain, ydev,  ytest = self.open_data(DIR)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest

    def to_array(self, integer, num_labels):
        return np.array(np.eye(num_labels)[integer])

    def convert_ys(self, y):
        if 'negative' in y:
            return 0
        elif 'neutral' in y:
            return 1
        elif 'objective' in y:
            return 1
        elif 'positive' in y:
            if self.binary:
                return 1
            else:
                return 2

    def open_data(self, DIR):
        train = []
        tknzr = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)
        for line in open(os.path.join(DIR, 'train.tsv')):
            idx, sidx, label, tweet = line.split('\t')
            if self.binary:
                if 'neutral' in label or 'objective' in label:
                    pass
                else:
                    train.append((label, tweet))
            else:
                train.append((label, tweet))

        dev = []
        for line in open(os.path.join(DIR, 'dev.tsv')):
            idx, sidx, label, tweet = line.split('\t')
            if self.binary:
                if 'neutral' in label or 'objective' in label:
                    pass
                else:
                    dev.append((label, tweet))
            else:
                dev.append((label, tweet))

        test = []
        for line in open(os.path.join(DIR, 'test.tsv')):
            idx, sidx, label, tweet = line.split('\t')
            if self.binary:
                if 'neutral' in label or 'objective' in label:
                    pass
                else:   
                    test.append((label, tweet))
            else:
                test.append((label, tweet))

        ytrain, Xtrain = zip(*train)
        ydev,   Xdev   = zip(*dev)
        ytest,  Xtest  = zip(*test)
                    
        Xtrain = [rem_urls(tknzr.tokenize(sent.lower())) for sent in Xtrain]
        ytrain = [self.convert_ys(y) for y in ytrain]

        Xdev = [rem_urls(tknzr.tokenize(sent.lower())) for sent in Xdev]
        ydev = [self.convert_ys(y) for y in ydev]

        Xtest = [rem_urls(tknzr.tokenize(sent.lower())) for sent in Xtest]
        ytest = [self.convert_ys(y) for y in ytest]

        if self.one_hot_label:
            if self.binary:
                ytrain = [self.to_array(y, 2) for y in ytrain]
                ydev = [self.to_array(y, 2) for y in ydev]
                ytest = [self.to_array(y, 2) for y in ytest]
            else:
                ytrain = [self.to_array(y, 3) for y in ytrain]
                ydev = [self.to_array(y, 3) for y in ydev]
                ytest = [self.to_array(y, 3) for y in ytest]

        Xtrain = np.array(Xtrain)
        Xdev = np.array(Xdev)
        Xtest = np.array(Xtest)

        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)
        
        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

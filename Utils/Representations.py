import numpy as np
import re


def pooling(sentence, model, type):
    sent = []
    for w in sentence:
        if w in model:
            sent.append(model[w])

    if len(sent) == 0:
        for k, v in model.items():
            return np.zeros(len(v))

    sent = np.array(sent)
    if type == 'max':
        return np.max(sent, 0)
    elif type == 'min':
        return np.min(sent, 0)
    else:
        return np.average(sent, 0)


def clean_split(sent):
    s = sent.lower();
    s = re.sub(r'(https|http|file|ftp):\/\/[^ ]*( |$)|[#@][^ ]*( |$)', ' ', s)
    s = re.sub('[\W_]', ' ', s)
    s = re.sub('[\d]+', ' ', s)
    s = re.sub(' +', ' ',s);
    s = s.strip()
    return [w for w in s.split() if len(w) > 1]

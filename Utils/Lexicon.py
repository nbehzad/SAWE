import numpy as np
import re


def read_subjectivity_clues():
    sc_lexicon = {}
    with open('../lexicons/Subjectivity-clues.tff') as f:
        for line in f:
            entry = line.split(' ')
            strength = entry[0].split('=')[1]
            word = entry[2].split('=')[1]
            polarity = entry[len(entry) - 1].split('=')[1].strip()

            score = 0.0
            if polarity == 'positive' and strength == 'strongsubj':
                score = 8.0
            elif polarity == 'positive' and strength == 'weaksubj':
                score = 6.5
            elif polarity == 'negative' and strength == 'weaksubj':
                score = 3.5
            elif polarity == 'negative' and strength == 'strongsubj':
                score = 2.0
            elif polarity == 'neutral' or polarity == 'both':
                score = 5.0
            else:
                print('out of range entry for word %s' % word)

            if word in sc_lexicon:
                sc_lexicon[word] = (score + sc_lexicon[word]) / 2
            else:
                sc_lexicon[word] = score

    return sc_lexicon


def read_ANEW():
    ANEW_lexicon = {}
    with open('../lexicons/E-ANEW.csv') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
            else:
                entry = line.split(',')
                ANEW_lexicon[entry[1]] = float(entry[2])
    return ANEW_lexicon

def generate_final_lexicon(ANEW_lexicon, sc_lexicon):
    final_lexicon = {}
    common = 0
    for k, v in ANEW_lexicon.items():
        if k in sc_lexicon:
            final_lexicon[k] = (v + sc_lexicon[k]) / 2
            common += 1
        else:
            final_lexicon[k] = v

    sc_not_common = 0
    for k, v in sc_lexicon.items():
        if k not in final_lexicon:
            final_lexicon[k] = v
            sc_not_common += 1

    print('The final lexicon size = %d' % len(final_lexicon))
    print('The number of common words = %d' % common)
    print('The number of words in SC but not in ANEW lexicon = %d' % sc_not_common)

    return final_lexicon


def save_lecixon(lexicon):
    with open('../lexicons/final-lexicon', 'w') as f:
        for k, v in lexicon.items():
            f.write(k + ' :: ' + '{:.2f}'.format(v) + '\n')

final = generate_final_lexicon(ANEW_lexicon=read_ANEW(), sc_lexicon=read_subjectivity_clues())
save_lecixon(final)

import numpy as np


def read_vectors(file, vocab):
    """In some cases the first line of the file is
            the vocabulary length and vector dimension."""
    word2vec = {}
    vec_dim = 0
    embedding_length = 0
    first_line = True

    with open(file, encoding='utf8') as f:
        index = 0
        for line in f:
            if first_line:
                if line[0].isdigit() and len(line.split(' ')) < 3:
                    embedding_length, vec_dim = (int(p) for p in line.split(' '))
                else:
                    split = split = line.strip().split(' ')
                    vec_dim = len(split) - 1
                    word, vec = split[0].strip(), np.array(split[1:], dtype=float)
                    if word in vocab:
                        word2vec[word] = vec

                first_line = False
            else:
                split = line.strip().split(' ')
                word, vec = split[0].strip(), np.array(split[1:], dtype=float)

                if len(vec) == vec_dim:
                    if word in vocab:
                        word2vec[word] = np.array(vec, dtype=float)
                else:
                    print('The line %d is not a vector. The word is %s and its vec size is %d'
                          % (index + 1, word, len(vec)))

            index += 1

        print('The total lines in the embedding file is %d and the vocab length is %d' % (index, embedding_length))

    return word2vec, vec_dim

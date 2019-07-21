from keras.models import Sequential
from keras.models import Model
from keras.layers import Activation, Dense, Input
from keras import regularizers
from keras.optimizers import SGD
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


def get_w2v_model(source_file=None, vocab=None):
    w2v_model = {}

    if source_file is None:
        w2v = gensim.models.KeyedVectors.load_word2vec_format('./embeddings/GoogleNews-vectors-negative300.bin.gz'
                                                              , binary=True)
        if vocab is not None:
            for k, v in vocab.items():
                if k in w2v:
                    w2v_model[k] = w2v[k]
    else:
        w2v_model = get_glove_model(source_file=source_file, vocab=vocab)

    if vocab is not None:
        print('From %d words in vocab, %d words were matched with input w2v embedding' % (len(vocab), len(w2v_model)))
        return w2v_model
    else:
        return w2v


def get_glove_model(source_file=None, vocab=None):
    glove_model = {}
    if source_file is None:
        path = './embeddings/glove.840B.300d.txt'
    else:
        path = source_file

    with open(path) as f:
        if vocab is None:
            for line in f:
                word = line[0:line.index(' ')]
                strvec = line[line.index(' ') + 1:]
                glove_model[word] = [float(x) for x in strvec.split(' ')]
        else:
            for line in f:
                word = line[0:line.index(' ')]
                if word in vocab:
                    strvec = line[line.index(' ') + 1:]
                    glove_model[word] = [float(x) for x in strvec.split(' ')]
    if vocab is not None:
        print('From %d words in vocab, %d words were matched with input embedding' % (len(vocab), len(glove_model)))

    return glove_model


def get_vocab(input_vocab_file):
    vocab = {}
    with open(input_vocab_file) as file:
        for line in file:
            w, frq = line.split(' :: ')
            vocab[w] = int(frq.strip())

    return  vocab


def create_train_set(lexicon_file='./lexicons/final-lexicon', input_vocab_file=None, source_model='glove'
                     , source_embedding_file=None):
    lexicon = {}
    vocab = {}
    em_source_model = {}

    print('Importing lexicon <%s>...' % lexicon_file)
    with open(lexicon_file) as file:
        for line in file:
            w, score = line.split(' :: ')
            lexicon[w] = float(score.strip())

    if input_vocab_file is not None:
        print('Importing vocabulary <%s>...' % input_vocab_file)
        vocab = get_vocab(input_vocab_file)
    else:
        vocab = None

    print('Importing source embedding <%s>...' % source_model)
    if source_model == 'glove':
        em_source_model = get_glove_model(source_embedding_file, vocab)
    else:
        em_source_model = get_w2v_model(source_embedding_file, vocab)

    print('Generating train set...')
    xtrain = []
    ytrain = []
    word2index = {}
    index2word = {}
    index = 0
    words_not_in_lexicon = 0
    matched_words = 0
    for k, v in em_source_model.items():
        word2index[k] = index
        index2word[index] = k
        xtrain.append(v)
        if k in lexicon:
            matched_words += 1
            ytrain.append(lexicon[k] / 10)
        else:
            words_not_in_lexicon += 1
            ytrain.append(0.5)

        index += 1

    print('%d words are not in lexicon' % words_not_in_lexicon)
    print('%d words are matched with lexicon' % matched_words)

    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)

    return xtrain, ytrain, word2index, index2word

def get_pca(word_embedding, n_components=2):
    pca = PCA(n_components=n_components)
    we_pca = pca.fit_transform(word_embedding)
    return np.array(we_pca)


def train_senti_embedding(target_dim, xtrain, ytrain):
    model = Sequential()
    model.add(Dense(target_dim, input_dim=len(xtrain[0]), init="normal",
                    kernel_regularizer=regularizers.l2(0.01),
                    bias_regularizer=regularizers.l2(0.01),
                    activation="tanh"))
    model.add(Dense(1, activation='sigmoid'))

    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)binary_crossentropy
    # model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])mean_squared_error

    model.compile('adam', loss="mean_absolute_error", metrics=["mae"])
    model.fit(xtrain, ytrain, nb_epoch=200, batch_size=1024, verbose=1)

    h = model.layers[0].get_weights()
    wh = np.array(h[0])
    bh = np.array(h[1])
    senti_embedding = np.dot(xtrain, wh) + bh
    senti_embedding = np.apply_along_axis(np.tanh, 0, senti_embedding)

    '''
    senti_embedding = np.apply_along_axis(np.tanh, 0, senti_embedding)

    y = model.layers[1].get_weights()
    wy = np.array(y[0])
    by = np.array(y[1])

    i = 0 
    for line in senti_embedding:
        senti_embedding[i] = wy[:,np.argmax(ytrain[i])] * senti_embedding[i]     
        i += 1
    '''

    return senti_embedding


def train_senti_aware_embedding(target_dim, xtrain, ytrain, aware=True, pca=False):
    senti_aware_embedding = train_senti_embedding(target_dim, xtrain, ytrain)
    if aware:
        senti_aware_embedding = np.hstack((xtrain, senti_aware_embedding))
        if pca:
            senti_aware_embedding = get_pca(senti_aware_embedding, xtrain.shape[1])
    return senti_aware_embedding


def get_embeddings_dict(embedding_matrix, word2index):
    embedding_dict = {}
    for w in word2index:
        embedding_dict[w] = embedding_matrix[word2index[w]]
    return embedding_dict


def plot_testPN(X_pca, word2index):
    colors = ['red', 'navy', 'turquoise', 'darkorange']
    pos_words = [w.strip() for w in open('./lexicons/pos-words', 'r')]
    neg_words = [w.strip() for w in open('./lexicons/neg-words', 'r')]

    pos_pca = []
    neg_pca = []

    for w in pos_words:
        if w in word2index:
            pos_pca.append(X_pca[word2index[w]])

    for w in neg_words:
        if w in word2index:
            neg_pca.append(X_pca[word2index[w]])

    pos_pca = np.array(pos_pca)
    neg_pca = np.array(neg_pca)

    target_names = ['positive', 'negative', 'both', 'objective']
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.scatter(pos_pca[:, 0], pos_pca[:, 1], color=colors[0], label=target_names[0], marker='o')
    plt.scatter(neg_pca[:, 0], neg_pca[:, 1], color=colors[1], label=target_names[1], marker='o')

    for label, x, y in zip(pos_words, pos_pca[:, 0], pos_pca[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(20, -10),
            textcoords='offset points', ha='right', va='bottom'
            # ,bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
            # ,arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
        )

    for label, x, y in zip(neg_words, neg_pca[:, 0], neg_pca[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(20, -10),
            textcoords='offset points', ha='right', va='bottom'
            # ,bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
        )

    plt.legend(loc=1)
    plt.show()


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_embedding(name='senti-word-embedding.txt', embedding=None, index2word=None):
    dir = './embeddings/senti-embedding/'
    create_directory(dir)
    with open(os.path.join(dir, name), 'w') as f:
        i = 0
        mat = np.matrix(embedding)
        for w_vec in mat:
            f.write(index2word[i] + " ")
            np.savetxt(f, fmt='%.6f', X=w_vec)
            i += 1
        f.close()
    print('Embeddings were successfully saved.')


def reduce_input_embeddings(vocab_file, embedding_name, embedding_file, output_file, isSave=True):
    vocab = get_vocab(vocab_file)
    if embedding_name == 'w2v':
        model = get_w2v_model(embedding_file, vocab)
    else:
        model = get_glove_model(embedding_file, vocab)

    if isSave:
        with open(output_file, 'w') as f:
            for k, v in model.items():
                f.write(k + " ")
                np.savetxt(f, fmt='%.6f', X=np.array(v, ndmin=2))
        print('The embeddings were successfully saved.')


def crate_w2v_sawe():
    sawe_names = ['sawe-tanh-conc-30-w2v.txt', 'sawe-tanh-conc-100-w2v.txt', 'sawe-tanh-pca-30-w2v.txt', 'sawe-tanh-pca-100-w2v.txt']

    xtrain, ytrain, word2index, index2word = create_train_set(source_embedding_file='./embeddings/w2v-reduced.txt')

    sawe = train_senti_aware_embedding(30, xtrain, ytrain, pca=False)
    save_embedding(sawe_names[0], sawe, index2word)

    sawe = train_senti_aware_embedding(100, xtrain, ytrain, pca=False)
    save_embedding(sawe_names[1], sawe, index2word)

    sawe = train_senti_aware_embedding(30, xtrain, ytrain, pca=True)
    save_embedding(sawe_names[2], sawe, index2word)

    sawe = train_senti_aware_embedding(100, xtrain, ytrain, pca=True)
    save_embedding(sawe_names[3], sawe, index2word)

#reduce_input_embeddings('./datasets/datasets-vocab','glove', './embeddings/glove.840B.300d.txt', './embeddings/glove-reduced.txt')
#reduce_input_embeddings('./datasets/datasets-vocab','w2v', None, './embeddings/w2v-reduced.txt')
crate_w2v_sawe()

'''
xtrain, ytrain, word2index, index2word = create_train_set(source_embedding_file='./embeddings/w2v-reduced.txt')
sawe = train_senti_aware_embedding(30, xtrain, ytrain, aware=True, pca=True)
save_embedding('sawe-tanh-pca-30-w2v-repeat.txt', sawe, index2word)

'''

import sys
import tabulate
import argparse
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from Utils.WordVector import *
from ClassificationModels.NN_Models import *
from Utils.SST import *
from Utils.Semeval import *
from Utils.Representations import *
from Utils.MyMetrics import *

DROPOUT = 0.5
DIM = 100
EPOCH_COUNT = 20
BATCH_SIZE = 50
LSTM_DIM = 168
VERBOSE = 1


def print_prediction(file, prediction):
    with open(file, 'w') as out:
        for line in prediction:
            out.write(str(line) + '\n')


def get_embedding_matrix(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def index_sent(sent, w2idx):
    return np.array([w2idx[w] for w in sent])


def convert_dataset(dataset, w2idx, maxlen=50):
    dataset._Xtrain = np.array([index_sent(s, w2idx) for s in dataset._Xtrain])
    dataset._Xdev = np.array([index_sent(s, w2idx) for s in dataset._Xdev])
    dataset._Xtest = np.array([index_sent(s, w2idx) for s in dataset._Xtest])
    dataset._Xtrain = pad_sequences(dataset._Xtrain, maxlen, padding='pre', truncating='post')
    dataset._Xdev = pad_sequences(dataset._Xdev, maxlen, padding='pre', truncating='post')
    dataset._Xtest = pad_sequences(dataset._Xtest, maxlen, padding='pre', truncating='post')
    return dataset


def pooling_dataset(dataset, vector_model):
    dataset._Xtrain = np.array([np.hstack((pooling(s, vector_model, 'min'), pooling(s, vector_model, 'max'),
                                pooling(s, vector_model, 'avg'))) for s in dataset._Xtrain])

    dataset._Xdev = np.array([np.hstack((pooling(s, vector_model, 'min'), pooling(s, vector_model, 'max'),
                                         pooling(s, vector_model, 'avg'))) for s in dataset._Xdev])

    dataset._Xtest = np.array([np.hstack((pooling(s, vector_model, 'min'), pooling(s, vector_model, 'max'),
                                          pooling(s, vector_model, 'avg'))) for s in dataset._Xtest])

    return dataset


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def run_classification(model_name, em_file):

    print('Importing datasets...')
    sst_fine = SST_Dataset('datasets/sst/', binary=False)
    sst_binary = SST_Dataset('datasets/sst', binary=True)
    semeval_binary = Semeval_Dataset('datasets/semeval', binary=True)
    semeval_ternary = Semeval_Dataset('datasets/semeval', binary=False)

    datasets = [semeval_binary, semeval_ternary, sst_binary, sst_fine]
    names = ['semeval_binary', 'semeval_ternary', 'sst_binary', 'sst_fine']

    results = []
    std_devs = []

    for name, dataset in zip(names, datasets):
        print('Running on %s...' % name)
        max_length = 0
        vocab = {}
        for sent in list(dataset._Xtrain) + list(dataset._Xdev) + list(dataset._Xtest):
            if len(sent) > max_length:
                max_length = len(sent)
            for w in sent:
                if w not in vocab:
                    vocab[w] = 1
                else:
                    vocab[w] += 1


        print('\nThe dataset %s splits: (train, dev, test) = (%d, %d, %d)' %
              (name, dataset._Xtrain.shape[0], dataset._Xdev.shape[0], dataset._Xtest.shape[0]))
        print('sentence max length is %d' % max_length)


        print('Importing word vectors...')
        vector_model, dim = read_vectors(em_file, vocab)

        wordvecs = {}
        unk = 0
        for w in vocab:
            if w in vector_model:
                wordvecs[w] = vector_model[w]
            else:
                unk += 1
                wordvecs[w] = np.random.uniform(-0.25, 0.25, dim)

        print('For dataset %s: Vocab size=%d and UNK size = %d ' % (name, len(vocab), unk))

        if not model_name.startswith('mlp'):
            embedding_matrix, word2index = get_embedding_matrix(wordvecs, k=dim)

            print('Converting and Padding dataset...')
            dataset = convert_dataset(dataset, word2index, max_length)
        else:
            print('min, max, and average pooling...')
            dataset = pooling_dataset(dataset, wordvecs)

        output_dim = dataset._ytest.shape[1]
        dataset_results = []
        for i, it in enumerate(range(1)):
            #np.random.seed()
            print(i + 1)
            base_dir = 'models/' + model_name + '/' + name + '/run'+ str(i+1)
            create_directory(base_dir)

            checkpoint = ModelCheckpoint(base_dir + '/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc',
                                         verbose=1, save_best_only=True, mode='auto')

            if model_name.startswith('cnn'):
                clf = create_cnn2(embedding_matrix, max_length, DROPOUT, output_dim)
            elif model_name.startswith('lstm'):
                clf = create_lstm(lstm_dim=LSTM_DIM, output_dim=output_dim, weights=embedding_matrix, dropout=DROPOUT)
            elif model_name.startswith('bilstm'):
                clf = create_bilstm(lstm_dim=LSTM_DIM, output_dim=output_dim, weights=embedding_matrix, dropout=DROPOUT)
            elif model_name.startswith('mlp'):
                clf = create_MLP(input_dim=dim*3, output_dim=output_dim, dropout=DROPOUT)
            
            h = clf.fit(dataset._Xtrain, dataset._ytrain, validation_data=[dataset._Xdev, dataset._ydev],
                        batch_size=BATCH_SIZE, epochs=EPOCH_COUNT, verbose=VERBOSE, callbacks=[checkpoint])



            weights = os.listdir(base_dir)
            best_val = 0
            best_weights = ''
            for weight in weights:
                val_acc = re.sub('weights.[0-9]*-', '', weight)
                val_acc = re.sub('.hdf5', '', val_acc)
                val_acc = float(val_acc)
                if val_acc > best_val:
                    best_val = val_acc
                    best_weights = weight

            clf = load_model(os.path.join(base_dir, best_weights))
            pred = clf.predict(dataset._Xtest, verbose=1)

            if isinstance(pred, list):
                pred = pred[-1]
            classes = []
            try:
                classes = clf.predict_classes(dataset._Xtest, verbose=1)
            except:
                classes = [np.argmax(y, axis=0) for y in pred]

            prediction_dir = base_dir.replace('models', 'predictions')
            create_directory(prediction_dir)
            prediction_file = prediction_dir + '/pred.txt'
            print_prediction(prediction_file, classes)

            labels = sorted(set(dataset._ytrain.argmax(1)))
            if len(labels) == 2:
                average = 'binary'
            else:
                average = 'macro'

            '''
            # The official metric for semeval 2013 is macro-F over positive and negative classes
            if name == 'semeval' and dataset._ytest.shape[1] == 3:
                pn_y = np.hstack((np.array(dataset._ytest[:, 0], ndmin=2).transpose()
                                  , np.array(dataset._ytest[:, 2], ndmin=2).transpose()))
                pn_pred = np.hstack((np.array(pred[:, 0], ndmin=2).transpose()
                                     , np.array(pred[:, 2], ndmin=2).transpose()))
                mm = MyMetrics(pn_y, pn_pred, labels=[0, 1], average=average)
            else:
                mm = MyMetrics(dataset._ytest, pred, labels=labels, average=average)
            '''

            mm = MyMetrics(dataset._ytest, pred, labels=labels, average=average)
            acc, precision, recall, macro_f1 = mm.get_scores()
            dataset_results.append([acc, precision, recall, macro_f1])

        dataset_results = np.array(dataset_results)
        ave_results = dataset_results.mean(axis=0)
        std_results = dataset_results.std(axis=0)
        print(u'acc: {0:.3f} \u00B1{1:.3f}'.format(ave_results[0], std_results[0]))
        print(u'prec: {0:.3f} \u00B1{1:.3f}'.format(ave_results[1], std_results[1]))
        print(u'recall: {0:.3f} \u00B1{1:.3f}'.format(ave_results[2], std_results[2]))
        print(u'f1: {0:.3f} \u00B1{1:.3f}'.format(ave_results[3], std_results[3]))

        results.append(ave_results)
        std_devs.append(std_results)

    results.append(list(np.array(results).mean(axis=0)))
    std_devs.append(list(np.array(std_devs).mean(axis=0)))
    names.append('overall')

    return names, results, std_devs, dim


def run_experiment(model_name, em_file, output_dir):
    names, results, std_devs, dim = run_classification(model_name, em_file)
    rr = [[u'{0:.3f} \u00B1{1:.3f}'.format(r, s) for r, s in zip(result, std_dev)] for result, std_dev in zip(results, std_devs)]
    table_data = [[name] + result for name, result in zip(names, rr)]
    table = tabulate.tabulate(table_data, headers=['dataset', 'acc', 'prec', 'rec', 'f1'], tablefmt='simple', floatfmt='.3f')
    
    if output_dir:
        create_directory(output_dir)
        with open(os.path.join(output_dir, 'result.txt'), 'a') as f:
            f.write('\n')
            f.write('+++%s+++\n' % model_name)
            f.write(table)
            f.write('\n')


def multiple_glove_experiment(model_name):
    models = [model_name, model_name + '-sawe-conc30', model_name + '-sawe-conc100', model_name +
              '-sawe-pca30', model_name + '-sawe-pca100']
    output = './result/'
    embeddings = ['./embeddings/glove-reduced.txt',
                 './embeddings/senti-embedding/sawe-tanh-conc-30-glove.txt',
                 './embeddings/senti-embedding/sawe-tanh-conc-100-glove.txt',
                 './embeddings/senti-embedding/sawe-tanh-pca-30-glove.txt',
                 './embeddings/senti-embedding/sawe-tanh-pca-100-glove.txt']

    index = 0
    for em in embeddings:
        print('Experiment over %s' % em)
        run_experiment(models[index], em, output)
        index += 1


def multiple_w2v_experiment(model_name):
    models = [model_name, model_name + '-w2v-sawe-conc30', model_name + '-w2v-sawe-conc100', model_name +
              '-w2v-sawe-pca30', model_name + '-w2v-sawe-pca100']
    output = './result/'
    embeddings = ['./embeddings/w2v-reduced.txt',
                './embeddings/senti-embedding/sawe-tanh-conc-30-w2v.txt',
                 './embeddings/senti-embedding/sawe-tanh-conc-100-w2v.txt',
                 './embeddings/senti-embedding/sawe-tanh-pca-30-w2v.txt',
                 './embeddings/senti-embedding/sawe-tanh-pca-100-w2v.txt']
    index = 0
    for em in embeddings:
        print('Experiment over %s' % em)
        run_experiment(models[index], em, output)
        index += 1



def main(args):
    parser = argparse.ArgumentParser(description='test senti-embeddings on several datasets')
    parser.add_argument('-model', help='The classification model, it can be one of cnn, bilstm, lstm, svm, mlp', default='bilstm')
    parser.add_argument('-emb', help='location of embeddings', default='/home/behzad/Projects/c-word2vec/wiki-em/12-wiki-lex-r-w2v-sub04.txt')
    parser.add_argument('-output', help='output file for printing results', default='./result/')

    args = vars(parser.parse_args())
    model = args['model']
    model = 'mlp-RL-glove-repeat'
    embedding = args['emb']
    embedding = './c-word2vec/output/12-wiki-sst-sem-lex-r-w2v.txt'
    embedding = './embeddings/glove-reduced.txt'
    output = args['output']

    print('Experiment over %s' % embedding)
    run_experiment(model, embedding, output)


if __name__ == '__main__':
    args = sys.argv
    main(args)

    #multiple_glove_experiment('lstm-glove-semeval')
    #multiple_glove_experiment('bilstm-glove-semeval')
    #multiple_glove_experiment('cnn-keras-glove-semeval')
    #multiple_glove_experiment('mlp-RL-glove')

    #multiple_w2v_experiment('cnn-keras-w2v-semeval')
    #multiple_w2v_experiment('mlp-w2v-semeval')
    #multiple_w2v_experiment('lstm-w2v-semeval')
    #multiple_w2v_experiment('bilstm-w2v-semeval')




from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, Embedding, Convolution1D, MaxPooling1D, Flatten, Input, Bidirectional, LSTM,\
    Concatenate, Reshape, Conv2D, MaxPool2D
from keras import regularizers
from keras.optimizers import Adam


def create_cnn(weights, sequence_length, drop=0.5, output_dim=2):

    vocabulary_size = weights.shape[0]
    embedding_dim = weights.shape[1]
    filter_sizes = [3, 4, 5]
    num_filters = 100

    print('Keras-CNN model with WE dim %d' % embedding_dim)
    print('num_filters %d' % num_filters)
    print('dropout %.2f' % drop)
    print('output dim %d' % output_dim)
    print('matrix shape [%d * %d]' % (weights.shape[0], weights.shape[1]))

    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length,
                          weights=[weights], trainable=False)(inputs)
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=output_dim, kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01),
                   activation='softmax')(dropout)


    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if output_dim == 2:
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    return model


def create_lstm(lstm_dim=150, output_dim=2, dropout=.5, weights=None):

    print('LSTM model with NON-TRAINABLE WE dim %d' % weights.shape[1])
    print('lstm dim %d' % lstm_dim)
    print('dropout %.2f' % dropout)
    print('output dim %d' % output_dim)
    print('matrix shape [%d * %d]' % (weights.shape[0], weights.shape[1]))

    model = Sequential()
    model.add(Embedding(weights.shape[0], weights.shape[1], weights=[weights], trainable=False))
    model.add(LSTM(lstm_dim))
    model.add(Dense(output_dim, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)
                    , activation='softmax'))
    if output_dim == 2:
        model.compile('adam', 'binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile('adam', 'categorical_crossentropy',
                      metrics=['accuracy'])

    print(model.summary())
    return model


def create_bilstm(lstm_dim=150, output_dim=2, dropout=.5, weights=None):

    print('BiLSTM model with NON-TRAINABLE WE dim %d' % weights.shape[1])
    print('lstm dim %d' % lstm_dim)
    print('dropout %.2f' % dropout)
    print('output dim %d' % output_dim)
    print('matrix shape [%d * %d]' % (weights.shape[0], weights.shape[1]))

    model = Sequential()
    model.add(Embedding(weights.shape[0], weights.shape[1], weights=[weights], trainable=False))
    model.add(Bidirectional(LSTM(lstm_dim)))
    model.add(Dense(output_dim, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)
                    , activation='softmax'))
    if output_dim == 2:
        model.compile('adam', 'binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile('adam', 'categorical_crossentropy',
                      metrics=['accuracy'])

    print(model.summary())
    return model


def create_bilstm_emtrainable(lstm_dim=150, output_dim=2, dropout=.5, weights=None):
    model = Sequential()

    print('BiLSTM model with TRAINABLE WE with dim %d' % weights.shape[1])
    print('lstm dim %d' % lstm_dim)
    print('dropout %.2f' % dropout)
    print('output dim %d' % output_dim)
    print('matrix shape [%d * %d]' % (weights.shape[0], weights.shape[1]))

    model.add(Embedding(weights.shape[0], weights.shape[1], weights=[weights], trainable=True))
    # model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(lstm_dim, dropout=dropout, recurrent_dropout=dropout)))
    model.add(Dropout(dropout))
    model.add(Dense(output_dim, kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)
                    , activation='softmax'))
    if output_dim == 2:
        model.compile('adam', 'binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile('adam', 'categorical_crossentropy',
                      metrics=['accuracy'])

    print(model.summary())
    return model


def create_LR(input_dim, output_dim=2, dropout=0.5):
    model_name = 'LR'
    print(model_name + ' model with feature size: %d, output: %d' % (input_dim, output_dim))

    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
    if output_dim == 2:
        model.compile('adam', 'binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile('adam', 'categorical_crossentropy',
                      metrics=['accuracy'])
    print(model.summary())
    return model

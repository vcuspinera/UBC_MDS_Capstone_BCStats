# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-21

'''This script will read the fasttext embedding matrix and padded documents for the training set, will 
fit the biGRU model, and save the model in the specified directory.

There are 2 parameters Input Path and Output Path where you want to save the trained model.

Usage: theme_train.py --input_dir=<input_file_path> --output_dir=<destination_dir_path>

Example:
    python src/models/theme_train.py --input_dir=data/interim/question1_models/advance --output_dir=models/Theme_Model/

Options:
--input_dir=<input_dir_path> Directory name for the padded documents and embeddings
--output_dir=<destination_dir_path> Directory for saving trained model
'''

import pandas as pd
import numpy as np

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate
from keras.layers import MaxPooling2D, GlobalMaxPooling1D, GRU, Bidirectional, GlobalAveragePooling1D
from keras.layers import Embedding, Input
from keras.layers.merge import concatenate
from keras import layers
import tensorflow as tf

import keras

from docopt import docopt

opt = docopt(__doc__)

def main(input_dir, output_dir):
    """
    Takes padded documents and embedding matrix from specified input_dir, trains biGRU 
    model and saves the trained model in specified output_dir
    """

    assert os.path.exists(input_dir), "The path entered for input_dir does not exist. Make sure to enter correct path \n"
    assert os.path.exists(output_dir), "The path entered for output_dir does not exist. Make sure to enter correct path \n"
    
    print("--- START: theme_train.py ---\n")
    print("**Reading the embedding matrix and padded documents**")
    ## Reading the embedding matrix (.npy file)
    embedding_matrix = np.load(input_dir + '/embedding_matrix.npy')

    ## Reading in the padded train document
    padded_doc_train = np.load(input_dir + '/X_train_padded.npy')

    ## Reading in y_train
    y_train = np.load(input_dir + '/y_train.npy')

    print("**Building the model now**")
    
    max_features = embedding_matrix.shape[0] ## vocabulary size
    maxlen = padded_doc_train.shape[1]
    embed_size = 300
    n_class = y_train.shape[1]

    epochs = 12
    batch_size = 100

    def define_model(length, max_features):
        """
        Model definition for Bi-GRU and fasttext model

        Parameters
        ----------
        length: (int) Maximum length among preprocessed comments
        max_features: (int) Vocabulary size of training comments

        Returns
        -------
        Tensorflow model
        """
        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inputs1)

        bi_gru = Bidirectional(GRU(278, return_sequences=True))(embedding1)

        global_pool = GlobalMaxPooling1D()(bi_gru)
        avg_pool = GlobalAveragePooling1D()(bi_gru)

        concat_layer = Concatenate()([global_pool, avg_pool])

        output = Dense(n_class, activation='sigmoid')(concat_layer)

        model = Model(inputs1, output)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy'])
        model.summary()
        return model
    
    model_bigru = define_model(maxlen, max_features)

    print("**Training the model. This will take some time, please be patient. (Note that it is not advisable to run this model on local system)**")

    model_bigru.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_bigru.fit(padded_doc_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    print("**Model has been successfully trained! Saving the trained model now. Thanks for your patience!**")
    model_bigru.save(output_dir + '/theme_model')

    print("--- END: theme_train.py ---")

if __name__ == "__main__":
    main(opt["--input_dir"], opt["--output_dir"])

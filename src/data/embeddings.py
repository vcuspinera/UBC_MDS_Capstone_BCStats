# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-10

'''This script load datasets, perform preprocessing on the comments, 
get the embedding matrix and padded dataset for train, validation and
test data.

Usage: src/data/embeddings.py --model=<model> --level=<level> --label_name=<label_name> --include_test=<include_test>

Example:
    python src/data/embeddings.py --model='fasttext' --level='theme' --label_name='' --include_test='False'
    python src/data/embeddings.py --model='fasttext' --level='subtheme' --label_name='SP' --include_test='True'
'''

import pandas as pd
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import codecs
from docopt import docopt

# Preprocess the data
import sys
sys.path.append('src/data/')
from preprocess import Preprocessing


opt = docopt(__doc__)

def main(model, level, label_name, include_test):
    # tests
    subthemes = ['CPD', 'CB', 'EWC', 'Exec', 'FWE',
        'SP', 'RE', 'Sup', 'SW', 'TEPE', 'VMG', 'OTH']
    
    if model not in ["fasttext", "glove"]:
        raise TypeError("The model options are 'fasttext' or 'glove'.\n")

    if level not in ["theme", "subtheme"]:
        raise TypeError("The level options are 'theme' or 'subtheme'.\n")
    
    if (level == "subtheme" and label_name not in subthemes):
        raise TypeError('Use one theme among next options:\n' + str(subthemes) + '\n')
    
    assert str(include_test).lower() in ["true", "false"] , "Select 'True' or 'False' for include_test.\n"

    if (level == "theme" and label_name != ''):
        print("\n⚠️  Warning: Runing themes, but receive a label_name different from ''.")

    # running the embeddings    
    print("\n--- START: embeddings.py ---")
    include_test = True if str(include_test).lower() == "true" else False
    emb = Embeddings()
    emb.make_embeddings(model=model, level=level, label_name=label_name, include_test=include_test)
    print("--- END: embeddings.py ---\n")
    return

class Embeddings:
    #Class for doing the embeddings and padded datasets

    def load_data(self, level="theme", label_name="", include_test=False):
        """
        Auxiliar function used to load the datasets of themes or a 
        specific Sub-theme, preprocess the X datasets and calculate
        the basic parameters that will use in other functions of
        this class.

        Parameters
        -------------
        level : (str)
            options are 'theme' and 'subtheme'
        label_name : (str)
            code of the sub-theme
        include_test : (boolean)
            True/False option to include or not the test dataset
        
        Returns
        -------------
        Nothing
        
        Example
        -------------
        from embeddings import Embeddings
        model = Embeddings()
        model.load_data(level="subtheme", label_name="FWE")
        """
        # load data
        if level == "theme":
            self.root = 'data/interim/question1_models/advance/'
            self.root_q2 = 'data/interim/question2_models/'
            exten = '.xlsx'
        else:
            self.root = 'data/interim/subthemes/' + label_name + '/'
            exten = '_subset.xlsx'

        self.X_train = pd.read_excel(self.root + 'X_train' + exten)['Comment'].tolist()
        self.X_valid = pd.read_excel(self.root + 'X_valid' + exten)['Comment'].tolist()
        self.y_train = pd.read_excel(self.root + 'y_train' + exten)
        self.y_valid = pd.read_excel(self.root + 'y_valid' + exten)

        if include_test:
            self.X_test = pd.read_excel(self.root + 'X_test' + exten)['Comment'].tolist()
            self.y_test = pd.read_excel(self.root + 'y_test' + exten)

        if level == "theme":
            self.data_q2 = pd.read_excel(self.root_q2 + 'comments_q2' + exten)['Comment'].tolist()
            self.data_2015 = pd.read_excel(self.root + 'data_2015' + exten)['Comment'].tolist()

        print('Loading: files were sucessfuly loaded')

        # # checking loaded data is not empty
        assert len(self.X_train) > 0, 'no records in X_train'
        assert len(self.X_valid) > 0, 'no records in X_valid'
        assert len(self.y_train) > 0, 'no records in y_train'
        assert len(self.y_valid) > 0, 'no records in y_valid'


        print('Preprocess: this step could take time, please be patient')
        self.X_train = Preprocessing().general(self.X_train)
        self.X_valid = Preprocessing().general(self.X_valid)
        if include_test:
            self.X_test = Preprocessing().general(self.X_test)
        if level == "theme":
            self.data_q2 = Preprocessing().general(self.data_q2)
            self.data_2015 = Preprocessing().general(self.data_2015)

        # Get parameters
        self.max_len = max(len(comment.split()) for comment in self.X_train)
        self.vect=Tokenizer()
        self.vect.fit_on_texts(self.X_train)
        self.vocab_size = len(self.vect.word_index) + 1
        return


    def make_embeddings(self, model="fasttext", level="theme", label_name="", include_test=False):
        """
        Function that gets the embeddings and padding datasets
        for themes or a specific sub-theme using GloVe vector 
        or Fasttext embeddings, and save these files in the
        correspondent folder.

        Parameters
        -------------
        model : (str)
            options are 'fasttext' and 'glove'
        level : (str)
            options are 'theme' and 'subtheme'
        label_name : (str)
            code of the sub-theme
        include_test : (boolean)
            True/False option to include or not the test dataset
        
        Returns
        -------------
        Nothing
        
        Example
        -------------
        from embeddings import Embeddings
        model = Embeddings()
        model.make_embeddings(level="subtheme", label_name="FWE")
        """
        # Loading datasets
        self.load_data(level, label_name, include_test)

        # Loading the whole embedding into memory
        print('Load Embeddings: loading the whole embedding into memory')
        embeddings_index = dict()
        if model == "fasttext":
            f = codecs.open('data/fasttext/crawl-300d-2M.vec')
        elif model == "glove":
            f = open('data/glove/glove.6B.300d.txt')

        for line in f:
            values = line.split()#.rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        # Create a weight matrix for words in training docs
        print('Embeddings: creating a weight matrix for words using', model, "model")
        embedding_matrix = np.zeros((self.vocab_size, 300))
        for word, i in self.vect.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # Padding data
        print('Padding: now is time for padding the embedding matrices')
        encoded_docs_train = self.vect.texts_to_sequences(self.X_train)
        padded_docs_train = pad_sequences(encoded_docs_train, maxlen=self.max_len, padding='post')
        encoded_docs_valid = self.vect.texts_to_sequences(self.X_valid)
        padded_docs_valid = pad_sequences(encoded_docs_valid, maxlen=self.max_len, padding='post')
        if include_test:
            encoded_docs_test = self.vect.texts_to_sequences(self.X_test)
            padded_docs_test = pad_sequences(encoded_docs_test, maxlen=self.max_len, padding='post')
        if level == "theme":
            encoded_question2 = self.vect.texts_to_sequences(self.data_q2)
            padded_question2 = pad_sequences(encoded_question2, maxlen=self.max_len, padding='post')
            encoded_data_2015 = self.vect.texts_to_sequences(self.data_2015)
            padded_data_2015 = pad_sequences(encoded_data_2015, maxlen=self.max_len, padding='post')
            

        # Saving the embedding matrix
        print('Save: saving files in ', self.root, ' directory.')
        np.save(self.root + 'embedding_matrix', embedding_matrix)

        # Saving the padding X's datafiles
        np.save(self.root + 'X_train_padded', padded_docs_train)
        np.save(self.root + 'X_valid_padded', padded_docs_valid)
        
        if include_test:
            np.save(self.root + 'X_test_padded', padded_docs_test)

        # Saving the padding y's datafiles
        np.save(self.root + 'y_train', self.y_train.iloc[:,:12])
        np.save(self.root + 'y_valid', self.y_valid.iloc[:,:12])
        if include_test:
            np.save(self.root + 'y_test', self.y_test.iloc[:,:12])

        # Saving the padding for question 2 and question 1's 2015 data
        if level == "theme":
            np.save(self.root_q2 + 'comments_q2_padded', padded_question2)
            np.save(self.root + 'data_2015_padded', padded_data_2015)

        return


if __name__ == "__main__":
    main(opt["--model"], opt["--level"], opt["--label_name"], opt["--include_test"])

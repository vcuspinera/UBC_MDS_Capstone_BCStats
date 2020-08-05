# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-08

'''This script will read in Q1 training and validation data from the specified directory,
preprocess the text using preprocessing.py and build the TFID vectorizer matrizes to be used
in the baseline model. Run this at the root project directory. 

Usage: tf-idf_vectorizer.py --input_dir=<input_dir_path> --output_dir=<destination_dir_path>

Example:
    python src/features/tf-idf_vectorizer.py \
    --input_dir=data/interim/question1_models/ \
    --output_dir=data/interim/question1_models/basic/

Options:
--input_dir=<input_dir_path> Directory name with the files
--output_dir=<destination_dir_path> Directory for saving the vectorizer npy file
'''

#Load dependencies
import sys
sys.path.insert(1, '.')
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data.preprocess import Preprocessing 
import scipy.sparse

from docopt import docopt

opt = docopt(__doc__)

def main(input_dir, output_dir):

    assert os.path.exists(input_dir), "The path entered for input_dir does not exist. Make sure to enter correct path \n"
    assert os.path.exists(output_dir), "The path entered for output_dir does not exist. Make sure to enter correct path \n"

    print("\n--- START: tf-idf_vectorizer.py ---")
        
    # Reading in datasets 
    X_train_Q1 = pd.read_excel(input_dir + "/advance/X_train.xlsx") 
    X_valid_Q1 = pd.read_excel(input_dir + "/advance/X_valid.xlsx")
    X_test_Q1 = pd.read_excel(input_dir + "/advance/X_test.xlsx")   

    assert len(X_train_Q1) > 0, 'no records of X_train.xlsx'
    assert len(X_valid_Q1) > 0, 'no records of X_valid.xlsx'
    assert len(X_test_Q1) > 0, 'no records of X_test.xlsx'

    print("Preprocessor started")

    # Preprocess train, valid, and train sets
    X_train_Q1['preprocessed_comments'] = Preprocessing().general(X_train_Q1['Comment'])
    X_valid_Q1['preprocessed_comments'] = Preprocessing().general(X_valid_Q1['Comment'])
    X_test_Q1['preprocessed_comments'] = Preprocessing().general(X_test_Q1['Comment'])

    
    #Tfid Vectorizer Representation
    def tfidf_vectorizer(train, valid, test):
        """
        Fits the TfidVectorizer() on your preprocessed 
        X_train set and transforms on X validation set.
        Returns the matrixes.
        """
        tfid = TfidfVectorizer() 
        X_train = tfid.fit_transform(train)
        X_valid = tfid.transform(valid)
        X_test = tfid.transform(test)

        return X_train, X_valid, X_test
    
    #Vectorize X_train and convert Y_train to an array
    X_train, X_valid, X_test = tfidf_vectorizer(X_train_Q1['preprocessed_comments'].values.astype('U'),
                                                X_valid_Q1['preprocessed_comments'].values.astype('U'),
                                                X_test_Q1['preprocessed_comments'].values.astype('U'))
    
    #Saving matrixes 
    scipy.sparse.save_npz(output_dir + '/tfidf_X_train', X_train)
    scipy.sparse.save_npz(output_dir + '/tfidf_X_valid', X_valid)
    scipy.sparse.save_npz(output_dir + '/tfidf_X_test', X_test)

    print("TF-IDF matrixes created and saved in output directory")
    print("--- END: tf-idf_vectorizer.py ---\n")

if __name__ == "__main__":
    main(opt["--input_dir"], opt["--output_dir"])
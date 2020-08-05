# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-05

'''This script will read and predicted label values for test comments from the theme model from the specified directory 
and will predict the subthemes for these comments. The output dataframe will be saved it in specified directory.

There are 2 parameters Input Path and Output Path where you want to write the evaluations of the subtheme predictions.

Usage: predict_subtheme.py --input_dir=<input_file_path> --output_dir=<destination_dir_path>

Example:
    python src/models/predict_subtheme.py --input_dir=data/ --output_dir=reports/tables/subtheme_tables/

Options:
--input_dir=<input_dir_path> Directory name for the excel files
--output_dir=<destination_dir_path> Directory for saving split files
'''

import pandas as pd
import numpy as np
import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sys
sys.path.append('src/data/')
from preprocess import Preprocessing

import os
from docopt import docopt

opt = docopt(__doc__)

def main(input_dir, output_dir):
    """
    This function loads files from input_dir, makes subtheme predictions based on the saved models
    and saves an evaluations on test set in the output_dir
    """
    assert os.path.exists(input_dir), "The path entered for input_dir does not exist. Make sure to enter correct path \n"
    assert os.path.exists(output_dir), "The path entered for output_dir does not exist. Make sure to enter correct path \n"

    print("----START: predict_subtheme.py----\n")
    print("**Loading data and generating necessary dictionaries**")
    ## Reading the comment prediction (.npy file)
    theme_pred = np.load(input_dir + 'output/theme_predictions/theme_question1_test.npy')

    ## Reading in the input comments
    X_test = pd.read_excel(input_dir + 'interim/question1_models/advance/X_test.xlsx')
    assert len(X_test) > 0, 'no records in X_test.xlsx'

    ## Reading y_test
    y_test = pd.read_excel(input_dir + 'interim/question1_models/advance/y_test.xlsx')
    assert len(y_test) > 0, 'no records in y_test.xlsx'

    y_test_subthemes = y_test.iloc[:,12:-1]
    
    ## Creating dictionary with theme indices as keys predicted comment indices as values
    ind_dict = dict()
    for i in range(theme_pred.shape[1]):
        ind_dict[i] = np.where(theme_pred[:,i] == 1)[0]

    ## Creating 2d zero array of size (#comments x 62)
    zero_arrays = np.zeros((theme_pred.shape[0], 62))
    
    ## Creating dictionary for subtheme range of columns
    theme_names = y_test.rename(columns={'FEW':'FWE'}).iloc[:,:12].columns
    subthemes = y_test.iloc[:,12:-1].columns
    subtheme_pos = dict()

    count_i = 0

    for i in range(len(theme_names)):
        count_a = count_i
        for sublab in subthemes:
            if sublab.startswith(theme_names[i]):
                count_i += 1
        subtheme_pos[i] = range(count_a, count_i)
    
    ## Creating dictionary for theme names and theme indices
    theme_dict = dict()
    model_dict = dict()
    for i in range(len(theme_names)):
        model_dict[i] = str(theme_names[i]).lower() + '_model'
        theme_dict[i] = str(theme_names[i])
    
    pred_thresh = {0:0.4, 1:0.4, 2:0.3, 3:0.4, 4:0.5, 5:0.3, 6:0.4, 7:0.4, 8:0.4, 9:0.3, 10:0.3, 11:0.4}

    ## Loop for predicting subthemes
    pred_subthemes = {}
    for i in list(ind_dict.keys()):
    
        print("**Predicting subthemes for comments classified as label", theme_dict[i], "**")
        print("**Subsetting the comments data**")

	# subset comments for predicted label
        comments_subset = X_test.iloc[ind_dict[i]]

	# load respective train set for predicted label
        input_dir_1 = input_dir + '/interim/subthemes/' + str(theme_dict[i])
        x_train = pd.read_excel(input_dir_1 + '/X_train_subset.xlsx')

	# Preprocessing comments and x_train
        print("**Preprocessing X_test and training set for label. This may take a little time**")
        x_train = Preprocessing().general(x_train['Comment'])
        comments_subset = Preprocessing().general(comments_subset['Comment'])

	# Getting parameters
        print("**Getting the required parameters now!!**")
        max_len = max(len(comment.split()) for comment in x_train)
        vect=Tokenizer()
        vect.fit_on_texts(x_train)

	# Padding comments
        encoded_docs_comments = vect.texts_to_sequences(comments_subset)
        padded_docs_comments = pad_sequences(encoded_docs_comments, maxlen=max_len, padding='post')    


	# loading model
        print("**Loading saved model for theme", model_dict[i], "**")
        model = tf.keras.models.load_model(input_dir + '/../models/Subtheme_Models/' + model_dict[i])

	# Predictions
        print("**Predicting subthemes for comments**")
        pred = model.predict(padded_docs_comments)
        pred = (pred > pred_thresh[i])*1
        
        pred_subthemes[i] = pred
        for j in range(pred_subthemes[i].shape[0]):
            zero_arrays[ind_dict[i][j], subtheme_pos[i]] += pred_subthemes[i][j]  
        print("**Predictions for subthemes of ", theme_dict[i], "are completed!**")
        print('\n')

    accuracy = []
    precision = []
    recall = []
    subtheme_model = []
    f1 = []
    for i in range(12):
        subtheme_model.append(theme_dict[i])
        accuracy.append(accuracy_score(np.asarray(y_test_subthemes.iloc[:,subtheme_pos[i]]), zero_arrays[:,subtheme_pos[i]]))
        precision.append(precision_score(np.asarray(y_test_subthemes.iloc[:,subtheme_pos[i]]), zero_arrays[:,subtheme_pos[i]], average='micro'))
        recall.append(recall_score(np.asarray(y_test_subthemes.iloc[:,subtheme_pos[i]]), zero_arrays[:,subtheme_pos[i]], average='micro'))
        f1.append(f1_score(np.asarray(y_test_subthemes.iloc[:,subtheme_pos[i]]), zero_arrays[:,subtheme_pos[i]], average='micro'))
    
    results = pd.DataFrame(data={'Subtheme_model':subtheme_model, 'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, 'F1 Score':f1})
    results.to_csv(output_dir + 'subtheme_pred_results.csv')
    print("**Results of test set subtheme predictions are saved in", output_dir, "**")
    print("----END: predict_subtheme.py----")

if __name__ == "__main__":
    main(opt["--input_dir"], opt["--output_dir"])
# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-22

'''This script will read new comments file and will predict the themes and subthemes for the comments. 
The output dataframe will be saved in the specified directory.

There are 2 parameters Input Path and Output Path where you want to write the file with theme and subtheme predictions.

Usage: predict_new_comments.py --input_dir=<input_file_path> --output_dir=<destination_dir_path>

Example:
    python src/models/predict_new_comments.py --input_dir=data/new_data/ --output_dir=data/new_data/

Options:
--input_dir=<input_dir_path> Directory name for new comments excel file
--output_dir=<destination_dir_path> Directory for saving excel file with predicted themes and subthemes 
'''

import pandas as pd
import numpy as np
import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sys
sys.path.append('src/data/')
from preprocess import Preprocessing

from docopt import docopt

opt = docopt(__doc__)

def main(input_dir, output_dir):
    """
    This function loads files from input_dir, makes theme and subtheme predictions
    based on the saved models and saves an excel file with predictions in the output_dir
    """
    print("\n--- START: predict_new_comment.py ---\n")

    print("**Loading the data**")
    ## Reading new comments data
    try:
        new_comments = pd.read_excel(input_dir + '/new_comments.xlsx')
    except:
        print("File new_comments.xlsx not found.\n")
        print("--- END: predict_new_comments.py ---\n")
        return

    ## Load training data
    X_train = pd.read_excel('data/interim/question1_models/advance/X_train.xlsx')

    ## Load y_train and extract column names for themes and subthemes
    y_train = pd.read_excel('data/interim/question1_models/advance/y_train.xlsx')
    theme_names = y_train.rename(columns={'FEW':'FWE'}).iloc[:,:12].columns
    subthemes = y_train.iloc[:,12:-1].columns

    print('**Preprocessing: this step could take time, please be patient.**')
    X_train = Preprocessing().general(X_train['Comment'])
    new_comments_ppd = Preprocessing().general(new_comments['Comment'])
    new_comment_ppd_df = pd.DataFrame(new_comments_ppd, columns = ['Comment'])

    ## Get parameters
    print('**Computing the required parameters**')
    max_len = max(len(comment.split()) for comment in X_train)
    vect=Tokenizer()
    vect.fit_on_texts(X_train)

    encoded_new_comments = vect.texts_to_sequences(new_comments_ppd)
    padded_new_comments = pad_sequences(encoded_new_comments, maxlen=max_len, padding='post')

    ## Loading saved model
    print('**Loading the saved theme model**')
    theme_model = tf.keras.models.load_model('models/Theme_Model/theme_model')
    print("**Making the theme predictions**")
    pred_themes_array = theme_model.predict(padded_new_comments)
    pred_themes_array = (pred_themes_array > 0.4)*1

    ## Making dataframe of prediction
    pred_themes = pd.DataFrame(pred_themes_array, columns=theme_names)

    print("**Theme predictions are successfully done. Predicting subthemes now.**\n")

    ## Creating dictionary with theme indices as keys predicted comment indices as values
    ind_dict = dict()
    for i in range(pred_themes_array.shape[1]):
        ind_dict[i] = np.where(pred_themes_array[:,i] == 1)[0]

    ## Creating 2d zero array of size (#comments x 62)
    zero_arrays = np.zeros((pred_themes_array.shape[0], 62))

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

    ## Loop for predicting subthemes
    pred_subthemes = dict()
    pred_thresh = {0:0.4, 1:0.4, 2:0.3, 3:0.4, 4:0.5, 5:0.3, 6:0.4, 7:0.4, 8:0.4, 9:0.3, 10:0.3, 11:0.4}
    
    for i in list(ind_dict.keys()):
    
        print("**Predicting subthemes for comments classified as label", theme_dict[i], "**")

	    # subset comments for predicted label
        # print("comment_subsets\n", new_comments_ppd)
        comments_subset = new_comment_ppd_df.iloc[ind_dict[i]] ## MAY BE DOESN'T NEED ILOC

	    # load respective train set for predicted label
        input_dir_1 = 'data/interim/subthemes/' + str(theme_dict[i])
        x_train = pd.read_excel(input_dir_1 + '/X_train_subset.xlsx')

	    # Preprocessing comments and x_train
        print("**Preprocessing training set for this label. This may take a little time**")
        x_train = Preprocessing().general(x_train['Comment'])
        # comments_subset = Preprocessing().general(comments_subset['Comment'])

	    # Getting parameters
        print("**Getting the required parameters now**")
        max_len = max(len(comment.split()) for comment in x_train)
        vect=Tokenizer()
        vect.fit_on_texts(x_train)

	    # Padding comments
        encoded_docs_comments = vect.texts_to_sequences(comments_subset['Comment'])
        padded_docs_comments = pad_sequences(encoded_docs_comments, maxlen=max_len, padding='post')

	    # loading model
        print("**Loading saved model for theme", model_dict[i], "**")
        model = tf.keras.models.load_model('models/Subtheme_Models/' + model_dict[i])

	    # Predictions
        print("**Predicting subthemes for comments**")
        try:
            pred = model.predict(padded_docs_comments)
            pred = (pred > pred_thresh[i])*1
            pred_subthemes[i] = pred
            for j in range(pred_subthemes[i].shape[0]):
                zero_arrays[ind_dict[i][j], subtheme_pos[i]] += pred_subthemes[i][j]
        except:
            next
        print("Predictions for subthemes of ", theme_dict[i], "are completed!")
        print('-----------------------------------')

    print("**Subtheme predictions are successfully done**")
    subtheme_pred = pd.DataFrame(zero_arrays, columns=subthemes)

    final_pred = pd.concat([pd.Series(new_comments['Comment']), pred_themes, subtheme_pred], axis=1)
    final_pred.to_excel(output_dir + '/predictions.xlsx')
    print("**Predictions have been saved to", output_dir, "**\n")
    print("--- END: predict_new_comments.py ---\n")

    return

if __name__ == "__main__":
    main(opt["--input_dir"], opt["--output_dir"])
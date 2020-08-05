# try and add training data points versus F1 score graph

# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-05

'''This script will read X_test and predicted label values for test comments from the theme model from the specified directory 
and will predict the subthemes for these comments. The output dataframe will be saved it in specified directory.

There are 2 parameters Input Path and Output Path where you want to write the evaluations of the subtheme predictions.

Usage: predict_subtheme.py --input_file=<input_file> --output_dir=<destination_dir_path>

Example:
    python src/models/predict_theme.py --input_file='theme_question1_test' --output_dir=data/output/theme_predictions/
    python src/models/predict_theme.py --input_file='theme_question2' --output_dir=data/output/theme_predictions/
    python src/models/predict_theme.py --input_file='theme_question1_2015' --output_dir=data/output/theme_predictions/

Options:
--input_file String for which predictions needs to be made
--output_dir=<destination_dir_path> Directory for saving predictions
'''

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior()
from docopt import docopt

opt = docopt(__doc__)

def main(input_file, output_dir):
    """
    Takes the input_file and calls make_predictions class with 
    output_dir as arguments
    """
    if input_file not in ["theme_question1_test", "theme_question2", "theme_question1_2015"]:
        raise TypeError("The input_file options are 'theme_question1_test', 'theme_question2' or 'theme_question1_2015.\n")

    assert os.path.exists(output_dir), "The path entered for output_dir does not exist. Make sure to enter correct path \n"

    print("----START: predict_theme.py----")
    mp = make_predictions()
    mp.predict(input_file=input_file, output_dir=output_dir)
    print('Thanks for your patience, the predictions have been saved!\n')
    print("----END: predict_theme.py----")
    return

class make_predictions:
    def load_data(self, input_file="theme_question1_test"):
        """
        Loads data according to input_file argument

        Parameters
        ----------
        input_file: (str) (theme_question2, theme_question1_2015, default value: theme_question1_test)

        Returns
        -------
        numpy array/ pandas dataframe

        """
        if input_file == 'theme_question1_test':
            self.padded_docs = np.load('data/interim/question1_models/advance/X_test_padded.npy')
            self.output_name = 'theme_question1_test'
            self.y_test = pd.read_excel('data/interim/question1_models/advance/y_test.xlsx')
            assert len(self.y_test) > 0, 'no records in y_test'
            self.y_test = self.y_test.iloc[:,:12]
            self.y_train = pd.read_excel('data/interim/question1_models/advance/y_train.xlsx')
            assert len(self.y_train) > 0, 'no records in y_train'
            self.y_train = self.y_train.iloc[:,:12]
        elif (input_file == 'theme_question1_2015'):
            self.padded_docs = np.load('data/interim/question1_models/advance/data_2015_padded.npy')
            self.output_name = 'theme_question1_2015'
        else:
            self.padded_docs = np.load('data/interim/question2_models/comments_q2_padded.npy')
            self.output_name = 'theme_question2'
        print('\nLoading: files were sucessfuly loaded.')

    def themewise_results(self, Ytrue, Ypred, Ytrain):
        '''Calculate accuracies for theme classification
        Parameters
        ----------
        Ytrue : array of shape (n_obeservations, n_labels)
            Correct labels for the 12 text classifications
        Ypred : array of shape (n_obeservations, n_labels)
            Predicted labels for the 12 text classifications
        Returns
        -------
        overall_results : dataframes of overall evaluation metrics
        theme_results : dataframe of evaluation metrics by class
        '''
        # Calculate individual accuracies and evaluation metrics for each class
        labels = ['CPD', 'CB', 'EWC', 'Exec', 'FWE', 'SP', 'RE', 'Sup', 'SW',
                'TEPE', 'VMG', 'OTH']
        Y_count = []
        pred_count = []
        Y_count_train = []
        accuracies = []
        precision = []
        recall = []
        f1 = []
        for i in np.arange(Ytrue.shape[1]):
            Y_count.append(np.sum(Ytrue.iloc[:, i] == 1))
            pred_count.append(np.sum(Ypred[:, i] == 1))
            Y_count_train.append(np.sum(Ytrain.iloc[:, i] == 1))
            accuracies.append(accuracy_score(Ytrue.iloc[:, i], Ypred[:, i]))
            precision.append(precision_score(Ytrue.iloc[:, i], Ypred[:, i]))
            recall.append(recall_score(Ytrue.iloc[:, i], Ypred[:, i]))
            f1.append(f1_score(Ytrue.iloc[:, i], Ypred[:, i]))
        theme_results = pd.DataFrame({'Label': labels,
                                    'Y_count': Y_count,
                                    'Pred_count': pred_count,
                                    'Y_count_train' : Y_count_train,
                                    'Accuarcy': accuracies,
                                    'Precision': precision,
                                    'Recall': recall,
                                    'F1 Score': f1})
        return theme_results

    def predict(self, input_file, output_dir):
        """
        Predicts the themes depending on the input_file and saved results using the 
        output_dir
        """
        "Predicts the theme for comments based on input file"
	    # Loading padded document for prediction
        self.load_data(input_file)
	
	    #Loading the model
        theme_model = tf.keras.models.load_model('models/Theme_Model/theme_model')
	
	    #Predictions
        print("**Making the predictions**")
        pred = theme_model.predict(self.padded_docs)
        pred = (pred > 0.4)*1

        if input_file == 'theme_question1_test':
            accuracy = []
            precision = []
            recall = []
            accuracy = accuracy_score(self.y_test, pred)
            precision = precision_score(self.y_test, pred, average='micro')
            recall = recall_score(self.y_test, pred, average='micro')
            f1 = f1_score(self.y_test, pred, average='micro')
            
            results = pd.DataFrame(data={'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, 'F1 Score':f1}, index=['theme_test_results'])
            results.to_csv('reports/tables/theme_tables/theme_pred_test_results.csv', index=False)
            print("**Results of test set theme prediction on test set are saved in reports/tables**")

            theme_results = self.themewise_results(self.y_test, pred, self.y_train)
            theme_results.to_csv('reports/tables/theme_tables/themewise_test_pr.csv', index=False)
            print("**Theme wise precision, recall metrics for main label model are saved in reports/tables**")
	
	    #Saving predictions
        print("**Now saving the predictions**")
        np.save(output_dir + self.output_name, pred)

if __name__ == "__main__":
    main(opt["--input_file"], opt["--output_dir"])

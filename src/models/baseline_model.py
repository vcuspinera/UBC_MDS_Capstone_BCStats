# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-08

'''This script will read in the Tf-idf vectorizer maxtrixes and train a Classifier Chain model.
It will save the model to the specified output directory. A results table which includes 
accuracies for all data and precision, recall, and f1 scores for validation and test data is 
also created and saved in reports/tables/ directory. 
Run at the root of directory.

Usage: baseline_model.py --input_dir=<input_dir_path> --output_dir=<destination_dir_path>

Example:
    python src/models/baseline_model.py \
    --input_dir=data/interim/question1_models/ \
    --output_dir=models/

Options:
--input_dir=<input_dir_path> Directory name with data files
--output_dir=<destination_dir_path> Directory for saving model and results in a csv file
'''

#Load dependencies
import numpy as np
import pandas as pd
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle
import scipy.sparse
import os

from docopt import docopt

opt = docopt(__doc__)

def main(input_dir, output_dir):

    assert os.path.exists(input_dir), "The path entered for input_dir does not exist. Make sure to enter correct path \n"
    assert os.path.exists(output_dir), "The path entered for output_dir does not exist. Make sure to enter correct path \n"
    
    print("\n--- START: baseline_model.py ---")
    print("Baseline Model Started")
    
    # Reading in y datasets                           
    y_train_Q1 = pd.read_excel(input_dir + "/advance/y_train.xlsx") 
    y_valid_Q1 = pd.read_excel(input_dir + "/advance/y_valid.xlsx")
    y_test_Q1 = pd.read_excel(input_dir + "/advance/y_test.xlsx")

    assert len(y_train_Q1) > 0, 'no records in y_train.xlsx'
    assert len(y_valid_Q1) > 0, 'no records in y_valid.xlsx'
    assert len(y_test_Q1) > 0, 'no records in y_test.xlsx'

    
    #Read in tfidf vectorizers for X
    X_train = scipy.sparse.load_npz('data/interim/question1_models/basic/tfidf_X_train.npz')
    X_valid = scipy.sparse.load_npz('data/interim/question1_models/basic/tfidf_X_valid.npz')
    X_test = scipy.sparse.load_npz('data/interim/question1_models/basic/tfidf_X_test.npz')


    #Slice y to themes and subthemes
    subthemes_ytrain = y_train_Q1.loc[:, 'CPD_Improve_new_employee_orientation':'Unrelated'] #62
    subthemes_yvalid = y_valid_Q1.loc[:, 'CPD_Improve_new_employee_orientation':'Unrelated']
    subthemes_ytest = y_test_Q1.loc[:, 'CPD_Improve_new_employee_orientation':'Unrelated']

    themes_ytrain = y_train_Q1.loc[:, 'CPD': 'OTH'] #12
    themes_yvalid = y_valid_Q1.loc[:, 'CPD': 'OTH']
    themes_ytest = y_test_Q1.loc[:, 'CPD': 'OTH']

    #Removing columns that have all 0 as labels
    if np.any(np.sum(subthemes_ytrain, axis=0) == 0):
        subthemes_yvalid = subthemes_yvalid.drop(subthemes_yvalid.columns[np.where(np.sum(subthemes_ytrain, axis=0) == 0)[0]], axis=1)
        subthemes_ytest = subthemes_ytest.drop(subthemes_ytest.columns[np.where(np.sum(subthemes_ytrain, axis=0) == 0)[0]], axis=1)
        subthemes_ytrain = subthemes_ytrain.drop(subthemes_ytrain.columns[np.where(np.sum(subthemes_ytrain, axis=0) == 0)[0]], axis=1)

    #Create empty dict
    results_dict = []

    #Classifier Chain function
    def Classifier_Chain(ytrain, yvalid, ytest, base_model):
        """
        Fits a Classifier Chain Model with LinearSVC as base classifier 
        specifiying either themes or subthemes for Y.
        Returns a table of results with train, valid, test score, and 
        recall, precision, f1 scores for valid and test data. 
        """
        classifier_chain = ClassifierChain(base_model)
        model = classifier_chain.fit(X_train, ytrain)
        
        train = model.score(X_train, np.array(ytrain)) 
        valid = model.score(X_valid, np.array(yvalid)) 
        test = model.score(X_test, np.array(ytest)) 

        #validation scores
        predictions = model.predict(X_valid)
        recall = recall_score(np.array(yvalid), predictions, average = 'micro') 
        precision = precision_score(np.array(yvalid), predictions, average = 'micro') 
        f1 = f1_score(np.array(yvalid), predictions, average = 'micro')
        
        #test scores
        predictions_test = model.predict(X_test)
        recall_test = recall_score(np.array(ytest), predictions_test, average = 'micro') 
        precision_test = precision_score(np.array(ytest), predictions_test, average = 'micro') 
        f1_test = f1_score(np.array(ytest), predictions_test, average = 'micro')
        
        #All rounded to 3 decimal place
        case = {'Model': "TF-IDF + LinearSVC",
            'Train Accuracy': round(train, 3),
            'Validation Accuracy': round(valid, 3),
            'Test Accuracy': round(test, 3),
            'Valid Recall': round(recall,3),
            'Valid Precision': round(precision, 3),
            'Valid F1': round(f1, 3) ,
            'Test Recall': round(recall_test, 3),
            'Test Precision': round(precision_test, 3),
            'Test F1': round(f1_test, 3)}
    
        results_dict.append(case)
    
    #Theme model
    model1 = Classifier_Chain(themes_ytrain, themes_yvalid, themes_ytest, LinearSVC())

    print("Theme baseline model success")
    
    #Save results in dataframe
    df = pd.DataFrame(results_dict)
    df.to_csv('reports/tables/baseline_results.csv')
    print("Results table saved in reports/tables directory")
    
    #Saving models as pickle
    with open(output_dir + 'baseline_theme.pkl', 'wb') as f:
        pickle.dump(model1, f)
    print("Baseline Model saved")

    print("--- END: baseline_model.py ---\n")
    
if __name__ == "__main__":
    main(opt["--input_dir"], opt["--output_dir"])

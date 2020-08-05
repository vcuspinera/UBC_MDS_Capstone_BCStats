# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-21

'''This script will read the saved theme/subtheme model(s), padded validation sets and y validation sets for model evaluation, 
and will save the evaluation results in the specified directory.

There are 2 parameters Input Path and Output Path where you want to save the evaluation results.

Usage: model_evaluate.py --level='theme' --output_dir=<destination_dir_path>

Example:
    python src/models/model_evaluate.py --level='theme' --output_dir=reports/
    python src/models/model_evaluate.py --level='subtheme' --output_dir=reports/

Options:
--input_dir=<input_dir_path> Directory name for the padded documents and embeddings
--output_dir=<destination_dir_path> Directory for saving evaluated results
'''

import pandas as pd
import numpy as np
from docopt import docopt

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

opt = docopt(__doc__)

print("\n-----START: model_evaluate.py-----\n")

def main(level, output_dir):
    """
    Takes the input level and calls model_evaluate class with 
    output_dir as argument 
    """
    me = model_evaluate()
    me.get_evaluations(level=level, output_dir=output_dir)
    print('Thanks for your patience, the evaluation process has finished!\n')
    print('----END: model_evaluate.py----\n')
    return

class model_evaluate:
    # Loads data and evaluates saved theme model and subtheme models on validation set
    
    def eval_metrics(self, model_name, x_valid, y_valid, level='theme'):
        """
        Evaluates model results on different threshold levels and produces data table/
        precision recall curves

        Parameters
        -----------
        model_name: (TensforFlow Saved model)
        x_valid: (pandas dataframe) dataframe with validation comments
        y_valid: (numpy array) array with labels
        level: (string) Takes value 'theme' or 'subtheme' to evaluate accordingly

        Returns
        -------
        Pandas DataFrame or matplotlib plot
        dataframe with evaluation metrics including precision, recall, f1 score at
        different threshold values
        """
        pred_values = model_name.predict(x_valid)

        if level == 'theme':
            precision_dict = dict()
            recall_dict = dict()
            thresh_dict = dict()

            precision_dict["BiGRU + Fasttext"], recall_dict["BiGRU + Fasttext"], thresh_dict["BiGRU + Fasttext"] = precision_recall_curve(y_valid.ravel(), pred_values.ravel())

            labels = []
            labels = list(precision_dict.keys())

            plt.figure()
            plt.step(recall_dict['BiGRU + Fasttext'], precision_dict['BiGRU + Fasttext'], where='post', color='orange')

            plt.xlabel('Recall', fontsize=18)
            plt.ylabel('Precision', fontsize=18)
            plt.axhline(y=0.743643, xmin=0, xmax=0.71, ls='--', color="cornflowerblue")
            plt.axvline(x=0.705382, ymin=0, ymax=0.71, ls='--', color="cornflowerblue")
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(labels, loc=(1.01, .79), prop=dict(size=14))
            plt.title('Precision Recall Curves for best performing model', fontsize = 18)
            plt.savefig('reports/figures/pr_curve_valid_theme.png')

        # PRECISION & RECALL
        predictions_results = []

        thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for val in thresholds:
            pred=pred_values.copy()
            pred[pred>=val]=1
            pred[pred<val]=0

            accuracy = accuracy_score(y_valid, pred, normalize=True, sample_weight=None)
            precision = precision_score(y_valid, pred, average='micro')
            recall = recall_score(y_valid, pred, average='micro')
            f1 = f1_score(y_valid, pred, average='micro')
            case= {'Threshold': val,
                  'Accuracy': accuracy,
                  'Precision': precision,
                  'Recall': recall,
                  'F1-measure': f1}
            predictions_results.append(case)

        return pd.DataFrame(predictions_results)
    
    def get_evaluations(self, level, output_dir):
        """
        Evaluates models by using eval_metrics function
        """
        if level == 'theme':
            print("**Loading data**")
            x_valid = np.load('data/interim/question1_models/advance/X_valid_padded.npy')
            y_valid = np.load('data/interim/question1_models/advance/y_valid.npy')
            print("**Loading the saved theme model**")
            model = tf.keras.models.load_model('models/Theme_Model/theme_model')
            print("**Predicting on validation set using saved model and evaluating metrics**")
            results = self.eval_metrics(model_name = model, x_valid = x_valid, y_valid = y_valid)
            print("**Saving results**")
            results.to_csv(output_dir + '/tables/theme_tables/theme_valid_eval.csv')
            print("Evaluations saved to reports/")

        else:
            print("Loading data and evaluating the subthemes model on validation set")
            themes = ['CPD', 'CB', 'EWC', 'Exec', 'FWE',
            'SP', 'RE', 'Sup', 'SW', 'TEPE', 'VMG', 'OTH']

            for label in themes:
                print("****Label:", label, "****")
                print("**Loading data**")
                x_valid = np.load('data/interim/subthemes/' + str(label) + '/X_valid_padded.npy')
                # self.x_valids.append(x_valid)
                y_valid = np.load('data/interim/subthemes/' + str(label) + '/y_valid.npy')
                # self.y_valids.append(y_valid)
                print("**Loading the saved subtheme model**")
                model = tf.keras.models.load_model('models/Subtheme_Models/' + str(label).lower() + '_model')
                # self.models.append(model)
                print("**Predicting on validation set using saved model and evaluating metrics**")
                results = self.eval_metrics(model_name = model, x_valid = x_valid, y_valid = y_valid, level = 'subtheme')
                print("**Saving results**")
                results.to_csv(output_dir + '/tables/subtheme_tables' + str(label).lower() + '_valid_eval.csv')
                print("Process of subtheme", label, "model completed\n")
            print("Evaluations saved to reports/tables")

if __name__ == "__main__":
    main(opt["--level"], opt["--output_dir"])

# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-17

'''This script call subset_subtheme_data.py script , perform preprocessing on the comments, 
get the embedding matrix and padded dataset for train, validation and
test data.

Usage: src/data/subthemes.py --input_dir=<input_dir_path> --model=<model> --include_test=<include_test>

Example:
    python src/data/subthemes.py --input_dir='data/interim/question1_models/advance/' --model='fasttext' --include_test='True'
'''

import pandas as pd
import numpy as np
import os
import sys
from docopt import docopt

opt = docopt(__doc__)

def main(input_dir, model, include_test):

    include_test = True if str(include_test).lower() == "true" else False

    print("\n--- START: subthemes.py ---")
    print('Start: This process could take time, please be patient')
    
    # load data
    print('Subset: The first step is to load datasets and subsetting the data')
    X_train = pd.read_excel(input_dir + 'X_train.xlsx')
    X_valid = pd.read_excel(input_dir + 'X_valid.xlsx')

    y_train = pd.read_excel(input_dir + 'y_train.xlsx')
    y_valid = pd.read_excel(input_dir + 'y_valid.xlsx')

    if include_test == True:
        X_test = pd.read_excel(input_dir + 'X_test.xlsx')
        y_test = pd.read_excel(input_dir + 'y_test.xlsx')
    
    # save the subsetting datasets
    print('Save: saving subsetting files for Subthemes')
    sys.path.append('src/data/')
    from subset_subtheme_data import subset_data
    themes = ['CPD', 'CB', 'EWC', 'Exec', 'FWE', 'SP', 
              'RE', 'Sup', 'SW', 'TEPE', 'VMG', 'OTH']
    for t in themes:
        subset_data(t, X_train, y_train, 'train')
        subset_data(t, X_valid, y_valid, 'valid')
        subset_data(t, X_test,  y_test,  'test')
    
    # Embeddings for all themes
    sys.path.append('src/data/')
    # import embeddings
    for t in themes:
        print("\nCreating embeddings and padded datasets for subtheme: " + t)
        os.system("python src/data/embeddings.py --model='fasttext' --level='subtheme' --label_name=" + t + " --include_test='True'")
    print("\n--- END: subthemes.py ---\n")

if __name__ == "__main__":
    main(opt["--input_dir"], opt["--model"], opt["--include_test"])

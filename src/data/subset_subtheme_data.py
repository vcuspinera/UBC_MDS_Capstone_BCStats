# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-06

import pandas as pd
import numpy as np
import os
import pickle

def subset_data(label_name, X, y, dataset_type = "train"):
    """
    Subsets a dataser for the provided label of question 1 for 
    subtheme classification and saves these datasets.
    
    Parameters
    ----------
    label_name: (str)
        name of the label/main theme for which data has to be subsetted
    X: (Pandas dataframe)
        dataframe containing raw comments
    y: (Pandas dataframe)
        dataframe containing labels values
    dataset_type: (str)
        select among train/validation/test datasets
        
    Returns
    -------
    None

    Example:
    --------
    from subset_subtheme_data import subset_data
    # Simple example
    subset_data('OTH', X_train, y_train, 'train')
    # Example for all labels
    themes = ['CPD', 'CB', 'EWC', 'Exec', 'FWE', 'SP', 
              'RE', 'Sup', 'SW', 'TEPE', 'VMG', 'OTH']
    for t in themes:
        subset_data(t, X_train, y_train, 'train')
        subset_data(t, X_valid, y_valid, 'valid')
        subset_data(t, X_test, y_test, 'test')
    """
    label = str(label_name)

    # tests
    subthemes = ['CPD', 'CB', 'EWC', 'Exec', 'FWE',
        'SP', 'RE', 'Sup', 'SW', 'TEPE', 'VMG', 'OTH']
        
    if (label not in subthemes):
        raise TypeError('Use one theme among next options:\n' + str(subthemes) + '\n')

    # function
    try:
        dir_name = os.mkdir('data/interim/subthemes/' + label)
    except:
        pass
    
    with open('data/interim/subthemes/subtheme_dict.pickle', 'rb') as handle:
        subtheme_dict = pickle.load(handle)
    
    # dataset
    subset = pd.concat([X, y[subtheme_dict[label]]], axis=1)
    subset['remove_or_not'] = np.sum(subset.iloc[:,1:], axis=1)
    subset = subset[subset['remove_or_not'] != 0]
    subset.drop(columns='remove_or_not', inplace=True)
    
    X_subset = subset['Comment']
    X_subset.to_excel('data/interim/subthemes/' + label + '/X_' + dataset_type + '_subset.xlsx', index=False)
    
    y_subset = subset.iloc[:, 1:]
    y_subset.to_excel('data/interim/subthemes/' + label + '/y_' + dataset_type + '_subset.xlsx', index=False)
    
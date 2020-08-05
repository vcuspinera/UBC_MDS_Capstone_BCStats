# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-05

'''This script will read raw data from the specified directory and will merge and split it into 
training, validation and test data sets and saves it in specified directory.
There are 2 parameters Input and Output Path where you want to write this data.

Usage: merge_split_data.py --input_dir=<input_dir_path> --output_dir=<destination_dir_path>

Example:
    python src/data/merge_split_data.py --input_dir=data/raw/ --output_dir=data/interim/

Options:
--input_dir=<input_dir_path> Directory name for the excel files
--output_dir=<destination_dir_path> Directory for saving split files
'''

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from docopt import docopt

opt = docopt(__doc__)

def main(input_dir, output_dir):

    assert os.path.exists(input_dir), "The path entered for input_dir does not exist. Make sure to enter correct path \n"
    assert os.path.exists(output_dir), "The path entered for output_dir does not exist. Make sure to enter correct path \n"
    
    print("\n--- START: Merge_split_data.py ---")

    ### Reading data for question 1 (all four years) ###
    print("Loading datasets for question 1")

    # Reading WES 2013, question 1 and standardizing column headers
    data_2013 = pd.read_excel(input_dir + "/2013/WES2013 1st Qual Sample - Coded.xlsx", 
                         sheet_name='2013 1st Qual Sample',
                         skiprows=1)

    assert len(data_2013) > 0, 'no records in 2013/WES2013 1st Qual Sample - Coded.xlsx'

    data_2013.rename(columns={'_telkey':'Telkey',
                              'AQ3345_13':'Comment'}, inplace=True)
    data_2013['Year'] = 2013
    
    # Reading WES 2015, question 1 (has unlabeled data only)
    data_2015 = pd.read_excel(input_dir + "/2015/WES2015 1st Qual UNCODED.xlsx")
    assert len(data_2015) > 0, 'no records in 2013/WES2015 1st Qual UNCODED.xlsx'

    data_2015.rename(columns={'_telkey':'Telkey',
                          'Q3345_13':'Comment'}, inplace=True)
    data_2015['Year'] = 2015
    
    # Reading WES 2018, question 1 and standardizing column headers
    data_2018 = pd.read_excel(input_dir + "/2018/WES2018 1st Qual Coded - Final Comments and Codes.xlsx", 
                         sheet_name='2018 1st Qual',
                         skiprows=1)   ## change your path for data
    assert len(data_2018) > 0, 'no records in 2013/WES2018 1st Qual Coded - Final Comments and Codes.xlsx'
    data_2018.rename(columns={'_telkey':'Telkey',
                              'Q3345_13':'Comment'}, inplace=True)
    data_2018['Year'] = 2018

    # Reading WES 2020, question 1 and standardizing column headers
    data_2020 = pd.read_excel(input_dir + "/2020/WES2020 1st Qual Coded - Final Comments and Codes.xlsx", 
                         sheet_name='2020 1st Qual',
                         skiprows=1)
    assert len(data_2018) > 0, 'no records in 2013/WES2020 1st Qual Coded - Final Comments and Codes.xlsx'
    data_2020.rename(columns={'Q3345_13:   What one thing would you like your organization to focus on to improve your work environment?':'Comment'}, inplace=True)
    data_2020['Year'] = 2020
    
    # Correcting names to compile the databases for question 1
    data_2013.rename(columns = {"Tools_Equipment_Physical_Environment":'TEPE',
        "Vision_Mission_Goals":'VMG',
        "Other":'OTH',
        "Other comments":'OTH_Other_related',
        "Positive comments": "OTH_Positive_comments"}, inplace=True)

    data_2018.rename(columns = {'FWE':'FEW',
        'CPD_Improve_performance_management':'CPD_Improve_performance',
        'CB_Improve_benefits':'CB_Improve_medical',
        'Exec_Strengthen_quality_of_executive_leadership':'Exec_Strengthen_quality_of_executive_leaders',
        'FWE_Leading_Workplace_Strategies':'FWE_Improve_and_or_expand_Leading_Workplace_Strategies_LWS',
        'TEPE__Ensure_safety_and_security':'TEPE__Ensure_safety',
        'TEPE_Better_supplies_equipment':'TEPE_Provide_better_equipment',
        'TEPE_Better_furniture':'TEPE_Provide_better_furniture',
        'TEPE_Better_computer_hardware':'TEPE_Provide_better_hardware',
        'VMG_Assess_plans_priorities':'VMG_Assess_plans',
        'VMG_Improve_program_implementation':'VMG_Improve_program',
        'VMG_Public_interest_and_service_delivery':'VMG_Pay_attention_to_the_public_interest',
        'VMG_Keep_politics_out_of_work':'VMG_Remove_political_influence'
        }, inplace=True)
    
    # Concatenating data and basic cleaning for 2013, 2018, 2020 for question 1
    frames = [data_2020, data_2018, data_2013]
    data_all = pd.concat(frames)
    data_all = data_all.reset_index(drop=True)
    data_all.drop(index=np.where(data_all['Comment'].isnull())[0], inplace=True)
    data_all.fillna(0, inplace=True)
    
    # Splitting into training, validation and test sets for question 1
    print("Splitting datasets in test-validation-test for question 1")
    X = data_all['Comment']
    y = data_all.drop(['Telkey', 'Comment', 'Year'], axis=1)
    
    X_trainvalid, X_test, y_trainvalid, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_trainvalid, y_trainvalid, test_size=0.20, random_state=42)

    # Writing split files to output directory for question 1
    print("Saving data for question 1")
    data_all.to_excel(output_dir + '/question1_models/advance/labeled_data.xlsx', index=False)
    
    X_train.to_excel(output_dir + '/question1_models/advance/X_train.xlsx', index=False)
    y_train.to_excel(output_dir + '/question1_models/advance/y_train.xlsx', index=False)
    
    X_valid.to_excel(output_dir + '/question1_models/advance/X_valid.xlsx', index=False)
    y_valid.to_excel(output_dir + '/question1_models/advance/y_valid.xlsx', index=False)
    
    X_test.to_excel(output_dir + '/question1_models/advance/X_test.xlsx', index=False)
    y_test.to_excel(output_dir + '/question1_models/advance/y_test.xlsx', index=False)
    
    # Concatenating unlabled data for question 1 for app
    frames_unlabeled = [data_all[['Telkey', 'Comment', 'Year']], data_2015]
    comments_q1 = pd.concat(frames_unlabeled)
    
    comments_q1.to_excel(output_dir + '/question1_models/advance/comments_q1.xlsx', index=False)

    # Saving unlabeled data from 2015's question 1 for predictions
    data_2015.to_excel(output_dir + '/question1_models/advance/data_2015.xlsx', index=False)
    
    ### Reading data for question 2 (all years) ###
    
    # Reading WES 2015, question 2 (has unlabeled data only)
    print("Loading datasets for question 2")
    data_2015_2 = pd.read_excel(input_dir + "/2015/WES2015 2nd Qual UNCODED.xlsx")
    data_2015_2.rename(columns={'telkey':'Telkey',
                          'Q4981_11':'Comment'}, inplace=True)
    assert len(data_2015_2) > 0, 'no records in 2015/WES2015 2nd Qual UNCODED.xlsx'
    data_2015_2['Year'] = 2015
    
    # Reading WES 2018, question 2
    data_2018_2 = pd.read_excel(input_dir + '/2018/WES2018 2nd Qual Coded - Final Comments and Codes.xlsx', 
                     sheet_name='2018 2nd Qual Coded (All)')
    assert len(data_2018_2) > 0, 'no records in 2015/WES2018 2nd Qual Coded - Final Comments and Codes.xlsx'
    data_2018_2.rename(columns={'Q4981_11':'Comment'}, inplace=True)
    data_2018_2['Year'] = 2018
    
    # Reading WES 2020, question 2 (has unlabeled data only)
    data_2020_2 = pd.read_excel(input_dir + "/2020/WES2020 2nd Qual UNCODED.xlsx", 
                             sheet_name='WES2020 Q4981_11 UNCODED')
    data_2020_2.rename(columns={'_telkey':'Telkey',
                          'AQ4981_11':'Comment'}, inplace=True)
    assert len(data_2020_2) > 0, 'no records in 2020/WES2020 2nd Qual UNCODED.xlsx'
    data_2020_2['Year'] = 2020
    
    # Basic Cleaning for question 2
    data_2018_2.drop(data_2018_2.tail(1).index,inplace=True)
    data_2018_2.drop(index=np.where(data_2018_2['Comment'].isna())[0], inplace=True)
    data_2018_2.fillna(0, inplace=True)
    data_2015_2.drop(index=np.where(data_2015_2['Comment'].isna())[0], inplace=True)
    
    # # Splitting labelled data for question 2
    # X_2 = data_2018_2['Comment']
    # y_2 = data_2018_2.drop(['Telkey', 'Comment', '# of codes'], axis=1)
    
    # X_trainvalid_2, X_test_2, y_trainvalid_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.20, random_state=42)
    # X_train_2, X_valid_2, y_train_2, y_valid_2 = train_test_split(X_trainvalid_2, y_trainvalid_2, test_size=0.20, random_state=42)
    
    # # Writing split files to output directory for question 2
    # X_train_2.to_excel(output_dir + '/question2_models/X_train_2.xlsx', index=False)
    # y_train_2.to_excel(output_dir + '/question2_models/y_train_2.xlsx', index=False)
    
    # X_valid_2.to_excel(output_dir + '/question2_models/X_valid_2.xlsx', index=False)
    # y_valid_2.to_excel(output_dir + '/question2_models/y_valid_2.xlsx', index=False)
    
    # X_test_2.to_excel(output_dir + '/question2_models/X_test_2.xlsx', index=False)
    # y_test_2.to_excel(output_dir + '/question2_models/y_test_2.xlsx', index=False)
    
    # Concatenating unlabled data for question 2 for app
    print("Saving dataset for question 2")
    frames_q2 = [data_2015_2[['Telkey', 'Comment', 'Year']], data_2018_2[['Telkey', 'Comment', 'Year']], data_2020_2[['Telkey', 'Comment', 'Year']]]
    comments_q2 = pd.concat(frames_q2)
    
    comments_q2.to_excel(output_dir + '/question2_models/comments_q2.xlsx', index=False)
    
    print("--- END: Merge_split_data.py ---\n")
    
if __name__ == "__main__":
    main(opt["--input_dir"], opt["--output_dir"])

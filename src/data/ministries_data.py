# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-12

'''This script read miniestries' data from raw directory and comments
data from interim directory, join the information both databases for
both questions, and saves it in specified directory.
There are 2 parameters Input and Output Path where you want to write this data.

Usage: ministries_data.py --input_dir=<input_dir_path> --output_dir=<destination_dir_path>

Example:
    python src/data/ministries_data.py --input_dir=data/ --output_dir=data/interim/

Options:
--input_dir=<input_dir_path> Location of data Directory
--output_dir=<destination_dir_path> Directory for saving ministries files
'''

import numpy as np
import pandas as pd
import re

from docopt import docopt

opt = docopt(__doc__)

def main(input_dir, output_dir):

    ## Ministries data
    print("\n--- START: ministries_data.py ---")
    print("Loading ministries' data into memory")

    # QUAN 2018
    quan_2018 = pd.read_excel(input_dir + "/raw/2018/WES2018 Quant and Driver data.xlsx", 
                        sheet_name='2018 Quant and Driver data')
    quan_2018.rename(columns={'telkey':'Telkey'}, inplace=True)
    quan_2018['Year'] = 2018
    quan_2018 =  quan_2018[['Telkey','ORGANIZATION', 'ORGID']]

    # QUAN 2015
    quan_2015 = pd.read_excel(input_dir + "/raw/2015/WES2015 Quant and Driver data.xlsx", 
                        sheet_name='2015 Quant and Driver data')   ## change your path for data
    quan_2015.rename(columns={'telkey':'Telkey'}, inplace=True)
    quan_2015['Year'] = 2015
    quan_2015 =  quan_2015[['Telkey','ORGANIZATION', 'ORGID']]

    # QUAN 2020
    quan_2020 = pd.read_excel(input_dir + "/raw/2020/WES2020 Quant and Driver data.xlsx", 
                        sheet_name='2020 Quant and Driver data')   ## change your path for data
    quan_2020.rename(columns={'telkey':'Telkey',
                            'ORGANIZATION20': 'ORGANIZATION',
                            'ORGID20': 'ORGID'}, inplace=True)
    quan_2020['Year'] = 2020
    quan_2020 =  quan_2020[['Telkey','ORGANIZATION', 'ORGID']]

    # Put databases together
    frames = [quan_2015, quan_2018, quan_2020]
    data_all = pd.concat(frames).rename(columns={'ORGANIZATION': 'Ministry',
                                                 'ORGID': 'Ministry_id'})
    data_all=data_all.drop_duplicates(subset='Telkey')

    
    ## Question 1
    print("Merging question 1 and ministries' data")

    # loading data
    data_q1 = pd.read_excel(input_dir + "/interim/question1_models/advance/labeled_data.xlsx")

    #remove - in Telkey
    data_q1['Telkey']= data_q1['Telkey'].astype(str)

    #wrangle Telkey values in 2013 to align with rest
    data_q1['Telkey'][31080:]= data_q1['Telkey'].str.replace(r'^((?:\D*\d){6})(?=.+)', r'\1-')[31080:]

    # Left Joining - Question 1 + Ministries
    ministries_Q1 = pd.merge(left=data_q1, right=data_all, how='left', left_on='Telkey', right_on='Telkey')


    ## Question 1 unlabeled data 2015

    data_2015 = pd.read_excel(input_dir + "/interim/question1_models/advance/data_2015.xlsx")   ## change your path for data

    # Left Joining - Question 1's 2015 data + Ministries
    ministries_2015 = pd.merge(left=data_2015, right=data_all, how='left', left_on='Telkey', right_on='Telkey')


    ## Question 2 (Unsupervised comments)
    print("Merging question 2 and ministries' data")

    # loading data
    data_q2 = pd.read_excel(input_dir + "/interim/question2_models/comments_q2.xlsx")   ## change your path for data

    #Left join
    ministries_Q2 = pd.merge(left=data_q2, right=data_all, how='left', left_on='Telkey', right_on='Telkey')

    # checking required columns in files before writing them
    assert 'Comment' in ministries_Q1.columns, 'Required column Comment is not present in ministries_Q1'
    assert 'Year' in ministries_Q1.columns, 'Required column Year is not present in ministries_Q1'
    assert 'Ministry' in ministries_Q1.columns, 'Required Columns are not present in ministries_Q1'
    
    assert 'Comment' in ministries_2015.columns, 'Required column Comment is not present in ministries_2015'
    assert 'Year' in ministries_2015.columns, 'Required column Year is not present in ministries_2015'
    assert 'Ministry' in ministries_2015.columns, 'Required Columns are not present in ministries_2015'

    assert 'Comment' in ministries_Q2.columns, 'Required column Comment is not present in ministries_Q2'
    assert 'Year' in ministries_Q2.columns, 'Required column Year is not present in ministries_Q2'
    assert 'Ministry' in ministries_Q2.columns, 'Required Columns are not present in ministries_Q2'


    ## Saving Excel files
    print("Saving merged datasets")
    ministries_Q1.to_excel(output_dir + "/question1_models/advance/ministries_Q1.xlsx", index=False)
    ministries_2015.to_excel(output_dir + "/question1_models/advance/ministries_2015.xlsx", index=False)
    ministries_Q2.to_excel(output_dir + "/question2_models/ministries_Q2.xlsx", index=False)
    

    print("--- END: ministries_data.py ---\n")
    
if __name__ == "__main__":
    main(opt["--input_dir"], opt["--output_dir"])

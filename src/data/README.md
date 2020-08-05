**File description in this directory**

Following is a brief overview of the function of each script/subdirectory present in this directory in the order of their usage:

| Script | Usage |
|----------|--------|
|merge_split_data.py| Merges the raw data for different years and then splits the data into train, validation and split datasets for Question 1. Also, prepares raw data for Question 2 and App|
|subset_subtheme_data.py| Function for creating subset of training, validation and test according to subthemes for building subtheme models|
|preprocess.py| Class with function for preprocessing the text data|
|embeddings.py| Creates embedding matrices using fasttext and padded documents for theme and subtheme model training as well as for Question 2 and Question 1-2015 theme prediction|
|subtheme.py|Uses subset_subtheme_data.py and embeddings.py to create embedding matrices and padded documents for training subtheme models|
|ministries_data.py| Joins ministry names to comments data for Question 1 and 2 to be used in the RShiny App |
|merge_ministry_pred.py| Merges theme predictions made using model with files produced by ministries_data.py to be used in RShiny App|
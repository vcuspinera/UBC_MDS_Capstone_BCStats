**File description in this directory**

Following is a brief overview of the function of each script/subdirectory present in this directory:

| Script | Usage |
|----------|--------|
|baseline_model.py| Trains and saves baseline model for theme prediction on Question 1. Also, saves evaluation results on test set.|
|theme_train.py| Trains Bi-GRU model with fasttext embedding on Question 1 training comments and saves trained model.|
|subtheme_models.py| Trains subtheme models (Bi-GRU/ CNN) with fasttext embedding on Question 1 training comments and saves trained models.|
|model_evaluate.py| Evaluates results of trained theme and subtheme models on the validation set and saves output tables with evaluation metrics.|
|predict_theme.py| Predicts themes using main theme model on specified comment set (test, Question 2 comments) and saves the evaluations.|
|predict_subtheme.py|Predicts subthemes using subtheme models on test set and saves the evaluations. |
|predict_new_comments.py|Predicts themes and subthemes for new comments and saves the predicted results. |
<img src="reports/figures/logo.png" width="150" align = "right">

BC Stats Capstone project 2020
-----------------------------
*Text Analytics: Quantifying the Repsonses to Open-Ended Survey Questions*

[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors**: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera  
**Mentor**: Varada Kolhatkar  
**Partner**: [BC Stats](https://www2.gov.bc.ca/gov/content/data/about-data-management/bc-stats)

## About
BC Stats conducts the Work Environment Survey (WES) on BC Public Service’s ministries with the goal of identifying areas for improvement and understanding employee’s experiences in the working environment. Currently, the comments to the open-ended questions have been manually encoded. Given a large number of employees across their 26 ministries, hand-labelling comments is expensive and time-consuming. In this project we propose using natural language processing and machine learning classification techniques to automate the labelling of text responses with the goals of build an useful model to automate multi-label text classification and gather insight on trends by themes across Ministries.

## Report
The final report is available [here](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/blob/master/reports/Final_Report.pdf).

## Dependencies

|Python packages|R packages|
|--------|--------|
|altair|shiny|
|altair-transform|shinydashboard|
|docopt|RColorBrewer|
|Keras|shinyBS|
|Keras-Applications|tidyverse|
|Keras-Preprocessing|wordcloud|
|Markdown|SnowballC|
|matplotlib|tm|
|nltk|readxl|
|numpy|tidytext|
|numpydoc|textdata|
|pandas|tidyr|
|pyLDAvis|tokenizers|
|requests|igraph|
|scikit-learn|ggraph|
|spacy|magrittr|
|tensorflow|stringr|
|tensorflow-estimator|data.table|
|tensorflow-hub|Hmisc|
|textblob|sentimentr|
|tokenizers|rlang|
||shinycssloaders|

## **Usage**

### **Running recipe (recommended)**
To replicate the analysis, clone this GitHub repository, and follow the next steps:

1. Install the [dependencies](#dependencies) listed above. You can do so by running the following command from your terminal with this cloned project repository set as the root.
```
make requirements
```

2. Add a copy of the sensitive data in the `data/raw` folder, and [download the Fasttext pre-trained embeddings](https://fasttext.cc/docs/en/english-vectors.html) file `crawl-300d-2M.vec.zip`. Save the extracted embeddings file in `data/fasttext`.

3. Run the following command at the command line/terminal from the root directory of this project to prepare data for the models.
```
make ready_model
```

4. Upload the embedding matrix and padded datasets with termination `.npy` saved in `data/interim/question1_models/advance` ([link](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/tree/report_visualizations/data/interim/question1_models/advance)) to [Google Drive](https://www.google.ca/drive/). These files contain a vectorial representation of the words from the preprocesed comments, so they don't show any sensitive information.

⚠️**Warning:** don't upload any other file, it may contain sensitive information.

5. Open and run the models in [Google Colab](https://colab.research.google.com/). You can upload [this](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/blob/master/notebooks/final_model.ipynb) notebook to Colab for training the main theme model and [this](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/blob/master/notebooks/Subtheme_Models.ipynb) notebook to train the subtheme models.

6. Follow the instructions in the notebooks specified in previous step to run and save the models. Make sure to download the saved model(s) from your Google Drive and replace the trained models in `models/Theme_Model/` and `models/Subtheme_Models` directories.

7. Run the following command at the command line/terminal from the root directory of this project to make predictions on the validation and test set and render the final report.
```
make advance_evaluation
```

This entire process could take a couple of hours.

### **Running all model from one command (not recommended)**

An alternative for the previous steps is to run the following command at the command line/terminal from the root directory of this project.

⚠️**Warning:** This process would run the models on your local system instead of the cloud, will take several hours, and your computer may crash in the process:
```
make all
```

### **Running Rshiny App**

To only run the RShiny App, clone this GitHub repository, and run the following steps at the command line/terminal from the root directory of this project:

1. Install the [dependencies](#dependencies) by running 
```
make requirements
```

2. Run the following to prepare the data for the dashboard
```
make ready_dashboard
```

3. Run the following to launch the dashboard
```
make dashboard
```

**Note**: In case the raw data and model remain unchanged, executing `make ready_dashboard` once is enough. In order to launch the dashboard again, only running `make dashboard` will suffice.

### **Predicting themes and subthemes for new comments**

In order to predict the themes and subthemes for new comments, clone this GitHub repository, and run the following steps at the command line/terminal from the root directory of this project:

1. Install the [dependencies](#dependencies) by running 
```
make requirements
```
2. Run the following command to prepare the required data:
```
make ready_model
```

3. In the directory `data/new_data`, save an excel (.xlsx) file by the name `new_comments.xlsx` with the following format:

| Comment |
|----------|
|*comment_1*|
|*comment_2*|
|*comment_3*|
|*comment_4*|
|*comment_5*|

4. Run the following command from the root of the project repository:
```
make new_prediction
```
The predicted themes and subthemes will be saved in *data/new_data/predictions.xlsx*

### **Cleaning the repository**

To reset the repository to a clean state, with no intermediate or results files, run the following command at the command line/terminal from the root directory of this project:
```
make clean
```

To delete all sensitive data and pre-trained embeddings, run the following command at the command line/terminal from the root directory of this project:
```
make clean_confidential
```

### **Expected Run Times**

We have run the previous commands and following are the approximate expected run time:

|Command|Expected time|
|:----------|:--------------|
|`make ready_model` | 45-60 min. |
|`make advance_evaluation`| 10-20 min. |
|`make ready_dashboard`| 20-30 min.|
|`make dashboard`| 10 sec.|
|`make baseline_model`| 7 min. |
|`make new_prediction`|10-15 min for 10,000 comments|
|`make clean`|1 sec. |
|`make clean_confidential`|1 sec.|
|`make advance_model` |⚠️ Did not execute|
|`make all`|⚠️ Did not execute|

## Project Organization    

    ├── LICENSE
    ├── Makefile           <- Makefile with make commands make like `make dashboard`.
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── fasttext       <- Data from third party sources.
    │   ├── glove          <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been formed from raw data.
    │   ├── raw            <- The original, immutable data dump.
    │   ├── new_data       <- New data for theme and subtheme predictions.
    │   └── output         <- Theme predictions for test set, Question 2 comments saved.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details.
    │
    ├── models
    │   ├── Subtheme_Models <- Trained and serialized subtheme models.
    │   └── Theme_Model    <- Trained and serialized theme model.
    │
    ├── notebooks          <- Jupyter notebooks.                         
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── figures        <- Generated graphics and figures to be used in reporting.
    │   └── tables         <- General .csv files used for the report and presentation.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported.
    │
    ├── Makefile           <- Makes project reproducible.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   │
    │   ├── data           <- Scripts to generate training, validation and test data for models and 
    │   |   |                 App (More details in directory's README).
    │   │   ├── merge_split_data.py
    │   │   ├── subset_subtheme_data.py
    │   │   ├── preprocess.py
    │   │   ├── embeddings.py
    │   │   ├── subtheme.py
    │   │   ├── ministries_data.py
    │   │   └── merge_ministry_pred.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling (more details
    │   |   │                 in directory's README).
    │   │   └── tf-idf_vectorizer.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions (more details in directory's README).
    │   │   ├── baseline_model.py
    │   │   ├── theme_train.py
    │   │   ├── subtheme_models.py
    │   │   ├── model_evaluate.py
    │   │   ├── predict_theme.py
    │   │   ├── predict_subtheme.py
    │   │   └── predict_new_comments.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations 
    │       │                 for the RShiny App.
    │       ├── custom_functions.R
    │       ├── eda_plots.Rmd
    │       ├── eda_wordcloud.Rmd
    │       ├── server.R
    │       ├── ui.R
    │       ├── data_dictionary.md
    │       └── dummy.html
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


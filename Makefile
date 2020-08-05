.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

# Make file for BC Stats Capstone project 2020
# author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera
# date: 2020-06-05
# 
# This driver script split the data, create the embedding matrix and padded 
# files, run the models, and clean the repo. This script takes no arguments. 
#
# usage: make requirements
#						to install all Python and R pacakges for Analysis
#
# usage: make ready_model
#						to prepare data for the models
#
# usage: make ready_dashboard
#						to prepare data for the dashboard
#
# usage: make dashboard
#						to run Dashboard using R as a server
#
# usage: baseline_model
#						to run the baseline model TF-IDF + LinearSVC
#
# usage: advance_model
# 						to run the Deep Learning models with pre-trained embeddings
#
# usage: advance_evaluation
#						to run the evaluation with the advance model
#
# usage: new_prediction
#						prediction of new comments
#
# usage: make clean
#						to clean up all the intermediate files
#
# usage: make clean_confidential
#						to clean up all raw data and pre-trained embeddings
#
# usage: make all
#						to run all the analysis and load the app
#						WARNING: We don't recommend running `make all`.  
#						This process would run the models on your local system
#						instead of the cloud, will take several hours, and your
#						computer may crash in the process.


#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = 591_capstone_2020_bc_stats_mds
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: 
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$ Rscript -e 'install.packages("shiny", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("shinydashboard", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("RColorBrewer", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("shinycssloaders", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("shinyBS", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("tidyverse", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("wordcloud", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("SnowballC", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("tm", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("readxl", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("tidytext", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("textdata", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("tidyr", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("tokenizers", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("igraph", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("ggraph", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("magrittr", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("stringr", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("data.table", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("Hmisc", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("sentimentr", repos="http://cran.us.r-project.org")'
	$ Rscript -e 'install.packages("rlang", repos="http://cran.us.r-project.org")'
	

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Prepare data for the models
ready_model:
	python src/data/merge_split_data.py --input_dir=data/raw/ --output_dir=data/interim/
	python src/data/ministries_data.py --input_dir=data/ --output_dir=data/interim/
	python src/data/embeddings.py --model='fasttext' --level='theme' --label_name='' --include_test='True'
	python src/data/subthemes.py --input_dir='data/interim/question1_models/advance/' --model='fasttext' --include_test='True'

## Prepare the data for the App
ready_dashboard:
	python src/data/merge_split_data.py --input_dir=data/raw/ --output_dir=data/interim/
	python src/data/ministries_data.py --input_dir=data/ --output_dir=data/interim/
	python src/data/embeddings.py --model='fasttext' --level='theme' --label_name='' --include_test='True'
	python src/models/predict_theme.py --input_file='theme_question2' --output_dir=data/output/theme_predictions/    
	python src/models/predict_theme.py --input_file='theme_question1_2015' --output_dir=data/output/theme_predictions/
	python src/data/merge_ministry_pred.py --input_dir=data/ --output_dir=data/interim/

## Load the App
dashboard:
	R -e "shiny::runApp('src/visualization/', launch.browser=TRUE)"

## Runs the baseline model
baseline_model:
	python src/features/tf-idf_vectorizer.py --input_dir=data/interim/question1_models/ --output_dir=data/interim/question1_models/basic/
	python src/models/baseline_model.py --input_dir=data/interim/question1_models/ --output_dir=models/

## Runs the advance model
advance_model:
	python src/models/theme_train.py --input_dir=data/interim/question1_models/advance --output_dir=models/Theme_Model/
	python src/models/subtheme_models.py --input_dir=data/interim/subthemes --output_dir=models/Subtheme_Models/

## Runs the evaluations for the advance model
advance_evaluation:
	python src/models/model_evaluate.py --level='theme' --output_dir=reports/    
	python src/models/model_evaluate.py --level='subtheme' --output_dir=reports/
	python src/models/predict_theme.py --input_file='theme_question1_test' --output_dir=data/output/theme_predictions/
	python src/models/predict_subtheme.py --input_dir=data/ --output_dir=reports/tables/subtheme_tables/
	Rscript -e "library(rmarkdown);render('reports/Final_Report.Rmd', output_format = 'pdf_document')"

## Returns prediction of new comments
new_prediction:
	python src/models/predict_new_comments.py --input_dir=data/new_data/ --output_dir=data/new_data/

## Delete all compiled Python files
clean:	
	rm -r data/interim/question1_models/basic/*
	rm -r data/interim/question1_models/advance/*
	rm -r data/interim/question2_models/*
	rm -r reports/tables/subtheme_tables/*
	rm -r reports/tables/theme_tables/*
	find data/new_data/. -mindepth 1 ! -name *.md -delete
	find reports/tables/ -name "*.csv" -type f -delete
	find data/interim/subthemes/ -mindepth 1 -maxdepth 1 -type d -exec rm -r {} \;
	rm reports/Final_Report.pdf

## Delete all confidential files
clean_confidential:
	rm -r data/raw/*
	find data/fasttext/. -mindepth 1 ! -name *.md -delete
	find data/glove/. -mindepth 1 ! -name *.md -delete


## Run all the model
###### WARNING: We don't recommend running it. This process would run the models 
###### on your local system instead of the cloud, will take several hours, and 
###### your computer may crash in the process.
all:
	# Prepare the data
	python src/data/merge_split_data.py --input_dir=data/raw/ --output_dir=data/interim/   	
	python src/data/ministries_data.py --input_dir=data/ --output_dir=data/interim/   	
	python src/data/embeddings.py --model='fasttext' --level='theme' --label_name='' --include_test='True'   	
	python src/data/subthemes.py --input_dir='data/interim/question1_models/advance/' --model='fasttext' --include_test='True'
	# Run baseline model
	python src/features/tf-idf_vectorizer.py --input_dir=data/interim/question1_models/ --output_dir=data/interim/question1_models/basic/
	python src/models/baseline_model.py --input_dir=data/interim/question1_models/ --output_dir=models/
	# Run advance model
	python src/models/theme_train.py --input_dir=data/interim/question1_models/advance --output_dir=models/Theme_Model/
	python src/models/subtheme_models.py --input_dir=data/interim/subthemes --output_dir=models/Subtheme_Models/
	# Perform evaluations for the advance model
	python src/models/model_evaluate.py --level='theme' --output_dir=reports/    
	python src/models/model_evaluate.py --level='subtheme' --output_dir=reports/
	python src/models/predict_theme.py --input_file='theme_question1_test' --output_dir=data/output/theme_predictions/
	python src/models/predict_subtheme.py --input_dir=data/ --output_dir=reports/tables/subtheme_tables/
	Rscript -e "library(rmarkdown);render('reports/Final_Report.Rmd', output_format = 'pdf_document')"
	# Load the App
	python src/models/predict_theme.py --input_file='theme_question2' --output_dir=data/output/theme_predictions/    
	python src/models/predict_theme.py --input_file='theme_question1_2015' --output_dir=data/output/theme_predictions/
	python src/data/merge_ministry_pred.py --input_dir=data/ --output_dir=data/interim/
	R -e "shiny::runApp('src/visualization/', launch.browser=TRUE)"
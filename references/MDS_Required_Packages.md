# Packages Used in MDS 2019/2020
*Compiled by Anny Chih*

After following the MDS installation instructions, creating an ‘mds’ environment in miniconda, and installing Jupyter Notebook and Jupyter Lab in miniconda, the following packages still needed to be installed:

## Jupyter
1.	Jupyter RISE:
    - conda install -c damianavila82 rise
    - jupyter-nbextension install rise --py --sys-prefix
    - jupyter-nbextension enable rise --py --sys-prefix
2.	conda install -c conda-forge jupyterlab 
    - Updating Jupyterlab to 2.0
3.	jupyter nbextension enable --py widgetsnbextension

## Python Packages
1.	conda install numpy
2.	conda install pandas
3.	(for Jupyter Lab): conda install -c conda-forge altair vega_datasets jupyterlab
4.	(for Jupyter Notebook): conda install -c conda-forge altair vega_datasets notebook vega
5.	(needed to use transform_density) pip install --upgrade altair
6.	conda install scipy
7.	conda install scikit-learn
8.	conda install -c conda-forge matplotlib
9.	conda install psutil
10.	conda install networkx
11.	conda install -c conda-forge pulp
12.	conda install snakeviz
13.	conda install numba
14.	conda install seaborn
15.	conda install -c conda-forge pandas-profiling
16.	conda install pillow
17.	(toy_classifier) conda install -c pytorch torchvision
18.	conda install python-graphviz
19.	conda install -c anaconda sqlalchemy
20.	conda install -c anaconda psycopg2
21.	conda install xlrd 
22.	(Mike’s plot_classified) pip install git+git://github.com/mgelbart/plot-classifier.git
23.	conda install -c conda-forge ipython-autotime
24.	conda install py-xgboost
25.	conda install nomkl
26.	pip install lightgbm
27.	(for installing lightgbm) pip install pypi-install 
28.	brew install libomp
29.	(Within Jupyter notebook): !pip install xgboost 
30.	pip install git+git://github.com/mgelbart/plot-classifier.git
31.	pip install -U scikit-learn
    - to get the latest release of scikit-learn
32.	conda install docopt
33.	conda install nltk
34.	conda install -c conda-forge librosa
35.	pip install tensorflow
36.	conda install -c conda-forge imbalanced-learn 
37.	conda install -c conda-forge eli5 #for DSCI 573
38.	conda install -c conda-forge scikit-image  
39.	conda install -c conda-forge shap 
40.	pip install pyjags
41.	pip install -U cookiecutter
42.	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
43.	conda install -c conda-forge keras
44.	conda install genism 
45.	conda install spacy 
46.	conda install pyemd 
47.	conda install -c anaconda beautifulsoup4 
48.	conda install lxml # DSCI 525
49.	pip install gocept.pseudonymize 
50.	python -m nltk.downloader 'punkt' 
51.	conda install -c conda-forge pyldavis 
52.	conda install -c conda-forge spacy 
53.	python -m spacy download en_core_web_sm 

## R Packages
1.	Convertemp: devtools::install_github(“ttimbers/convertemp”)
2.	Gapminder dataset: install.packages(“gapminder”)
3.	NYC Flights dataset: install.packages(“nycflights13”)
4.	install.packages("styler")
5.	install.packages("reticulate")
6.	install.packages(“xaringan”)
7.	install.packages("geojsonio")
8.	install.packages(“ggthemes”)
9.	install.packages(“sf”)
10.	install.packages("cowplot")
11.	install.packages("ggridges")
12.	install.packages(“magick”)
13.	install.packages("janitor")
14.	install.packages("ggimage")
15.	install.packages(“infer”)
16.	install.packages(“gridExtra”)
17.	install.packages(“HistData”)
    - Not actually necessary; but was used in textbook for DSCI 561
18.	install.packages(“tree”)
19.	install.packages(c('dbplyr', 'RPostgres'))
20.	install.packages(“GGally”)
21.	install.packages(“rsample”)
22.	install.packages(“plotly”)
23.	library(devtools)
    - install_github(‘plotly/dashR’)
24.	install.packages('tictoc')
25.	install.packages("car")
26.	install.packages("docopt") 
27.	install.packages("caret")
28.	install.packages("MLmetrics") 
    - There is a binary version available but the source version is later:
    - Do you want to install from sources the package which needs compilation? (Yes/no/cancel) Yes
29.	install.packages("zoo")
30.	install.packages("kableExtra")
31.	install.packages("marmap")
32.	install.packages("mapproj")
33.	remotes::install_github(“yihui/knitr”)
34.	install.packages("robust")
35.	install.packages("here")
36.	install.packages("ggfortify")
37.	install.packages("VGAM") # Must install this using RStudio (Not Jupyter Notebook!)
38.	install.packages("plm")
39.	install.packages("mice")
40.	install.packages(c("devtools", "roxygen2", "testthat", "knitr")) 
41.	install.packages("rjags")
42.	install.packages("forecast")
43.	install.packages('covr')
44.	install.packages("gstat") 
45.	install.packages(“fsa”) 
46.	install.packages(“qqman”) 
47.	install.packages(“gap”) 
48.	install.packages(“effects”) 
49.	install.packages(“pwr”)
50.	install.packages(“frequency”) 

## Other
1.	pip install dash
    - pip install dash-bootstrap-components
    - pip install gunicorn
2.	conda install "ipywidgets=7.5"
3.	(mds) 
    - brew tap mongodb/brew
    - brew install mongodb-community@4.2
        - To run MongoDB as a macOS service: brew services start mongodb-community
4.	(mds) brew install mongodb/brew/mongodb-community-shell
5.	(mds) brew install hugo
6.	Added ```RETICULATE_PYTHON = ‘/opt/miniconda3/envs/mds/bin/python’ ```  
    to the .Renviron file by typing `usethis::edit_r_environ()` in the RStudio console
    - I also added ``` Sys.setenv(RETICULATE_PYTHON = ‘/opt/miniconda3/envs/mds/bin/python’)``` to the .Rprofile file after Tiffany posted to follow these instructions for DSCI 524: https://ubc-mds.github.io/py-pkgs/setup.html#rstudio-python-setup
7.	brew install pandoc-citeproc 
8.	docker pull ttimbers/makefile2graph 
9.	docker pull rocker/tidyverse 
10.	docker pull debian:stable
11.	brew install pkg-config
12.	brew install jags
 
## Added to .gitignore_global
- .ipynb_checkpoints/
- .DS_Store
- altair-data-*.json

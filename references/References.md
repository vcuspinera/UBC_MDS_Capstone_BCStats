# References
__Relevant and Useful Resources__  

Author: Carlina Kim, Karanpal Singh, Sukriti Trehan, Victor Cuspinera  
Partner: BC Stats | Mentor: Varada Kolhatkar  

### BC Stats
- [Proposal of Capstone Partner](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/blob/master/references/BC-Stats_proposal_Text_Analytics.md)
- Work Environment Survey (WES):
    - [General information](https://www2.gov.bc.ca/gov/content/data/statistics/government/employee-research/wes/) (official site)
    - [Survey Guide of Themes](https://www2.gov.bc.ca/assets/gov/data/statistics/government/wes/wes2018_driver_guide.pdf)
    - [Dashboard of 2018 Quantitative Report by Ministries](https://securesurveys.gov.bc.ca/ERAP/workforce-profiles)

### Reports and presentations of the capstone project
- [Main Repo](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds)
- Proposal:
    - [Presentation](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/blob/master/reports/BCStats_Proposal.pdf)
    - [Report](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/blob/master/reports/BCStats_Proposal_Report.pdf)
- Final:
    - [Presentation to UBC-MDS](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/blob/master/reports/Final_Presentation.pdf)
    - [Presentation to BC Stats](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/blob/master/reports/BCStats_Presentation.pdf)
    - [Report](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/blob/master/reports/Final_Report.pdf)

### Models
- Baseline model:
    - [Multi-label classification (definition)](https://en.wikipedia.org/wiki/Multi-label_classification) by Wikipedia
    - [TF-IDF (definition)](https://en.wikipedia.org/wiki/Tf–idf) by Wikipedia
    - [Classifier Chains and Binary Relevance](https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/)
    - [Binary relevance efficacy for multilabel classification.](https://link.springer.com/article/10.1007/s13748-012-0030-x), 2012 paper
    - [Classifier Chains for Multi-label Classification.](https://link.springer.com/chapter/10.1007/978-3-642-04174-7_17), 2009 paper
    - [Scikit-Multilearn, User Guide](http://scikit.ml/userguide.html)
    - [Scikit-Multilearn, Classifier Chains](http://scikit.ml/api/0.1.0/api/skmultilearn.problem_transform.cc.html#skmultilearn.problem_transform.ClassifierChain) using _skmultilearn.problem_transform.ClassifierChain()_
    - [Scikit-Multilearn, MEKA wrapper](http://scikit.ml/meka.html#) wrapper _from skmultilearn.ext import Meka_
    - [MEKA, Classifier Chain](http://waikato.github.io/meka/meka.classifiers.multilabel.CC/)
    - [HMC (hierarchical Multi-Label model)](http://proceedings.mlr.press/v80/wehrmann18a.html) shared by Nasim Taba
    - [Decision Tree Hierachical Multi-Classifier](https://github.com/davidwarshaw/hmc) in Python
- Deep Learning models:
    - [Keras for Multi-label Text Classification](https://medium.com/towards-artificial-intelligence/keras-for-multi-label-text-classification-86d194311d0e) in Medium
    - [Multi-Class Text Classification with LSTM](https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17) in Medium
    - [Text Classification Using Convolutional Neural Networks](https://youtu.be/8YsZXTpFRO0) by Lukas Biewald in YouTube
    - [BI-GRU for text classification](https://medium.com/@felixs_76053/bidirectional-gru-for-text-classification-by-relevance-to-sdg-3-indicators-2e5fd99cc341) in Medium
    - [Illustrated Guide to LSTM’s and GRU’s](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21) in Medium
- Preprocess:
    - [Spacy glossary](https://github.com/explosion/spaCy/blob/master/spacy/glossary.py)
    - [Universal part-of-speech (POS) tags](https://universaldependencies.org/u/pos/)
    - [Spacy's part-of-speech and dependency tags meaning](https://stackoverflow.com/questions/40288323/what-do-spacys-part-of-speech-and-dependency-tags-mean) at StackOverflow
- Pre-trained Embeddings:
    - [FastText](https://fasttext.cc)
    - [Glove (Global Vectors for Word Representation)](https://nlp.stanford.edu/projects/glove/)
    - Universal Sentence Encoder:
        - [Original paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46808.pdf) by Google Research team, 2018
        - [Text embedding](https://tfhub.dev/google/universal-sentence-encoder/4) by TensorFlow Hub
        - [Keras meets Universal Sentence Encoder](https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/)
        - [Tutorial of classification for Universal Sentence Encoder](https://github.com/kvarada/Tutorials/blob/master/notebooks/text-classification/1_0_vk_classification_universal_sentence_encoding.ipynb) by [Varada Kolhatkar](https://kvarada.github.io)
    - BERT:
        - [Original paper](https://arxiv.org/pdf/1810.04805.pdf) by Google AI Language
        - [Announcement as State-of-the-Art Pre-training for NLP](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) by Google AI
        - [BERT fine-tuning tutorial](http://mccormickml.com/2019/07/22/BERT-fine-tuning/) by Chris McCormick
        - [Multilabel classification using BERT](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d) in Medium
- Evaluation:
    - [Precision-Recall for Multilabel classification](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
    - [ROC v/s Precision Recall Curves for Imbalanced data](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)
- Dashboard:
    - [Text Mining with R](https://www.tidytextmining.com)
    - [Shiny dashboard docs](https://rstudio.github.io/shinydashboard/index.html)
    - [Sentiment Analysis with sentimentr](https://medium.com/@ODSC/an-introduction-to-sentence-level-sentiment-analysis-with-sentimentr-ac556bd7f75a) in Medium
- Others:
    - [Cookie cutter](http://drivendata.github.io/cookiecutter-data-science/)
    - [GPU-Accelerated Machine Learning on MacOS](https://towardsdatascience.com/gpu-accelerated-machine-learning-on-macos-48d53ef1b545)

### Project of last year (2019) by Aaron, Ayla and Fan
- [Main Repo](https://github.com/aaronquinton/mds-capstone-bcstats)
- [Proposal Presentation](https://github.com/aaronquinton/mds-capstone-bcstats/blob/master/reports/proposal_presentation.pdf)
- [Final Report](https://github.com/aaronquinton/mds-capstone-bcstats/blob/master/reports/BCStats_Final_Report.pdf)

### Additional Resources
- [Working with Bigger Data](https://ttimbers.github.io/starting_to_work_with_bigger_data/presentation/starting_to_work_with_big_data.html#/) by Tiffany Timbers
- [MDS Required Packages](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/blob/master/references/MDS_Required_Packages.md) compiled by Anny Chih
- [Git commands](https://github.com/UBC-MDS/591_capstone_2020_bc-stats-mds/blob/master/references/Git_commands.md) compiled by Victor Cuspinera

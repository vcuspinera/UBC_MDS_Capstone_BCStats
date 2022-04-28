# GloVe
In order to run the model we developed, you should download the GloVe embeddings file `crawl-300d-2M.vec.zip`, that is a 2 million word vectors trained on Common Crawl (600B tokens).

You can download it [manually here](https://nlp.stanford.edu/projects/glove/), or using `wget`^[To run this command, you should install previusly `wget`.] command running the next code from the main directory of this repository.

```
wget -O data/glove/glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
```

In both cases you shound un-zip the folder and move the `glove.6B.300d.txt` file to the folder located in `src/glove`.
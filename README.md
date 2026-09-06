# Movie Review Sentiment Analysis

A simple script that trains a sentiment model on IMDb reviews, 3 times,
using more data each time (1,000 → 10,000 → 39,000 samples).

## What it does

1. Loads the IMDb dataset (50,000 movie reviews).
2. Cleans the text with spaCy (removes stopwords, lemmatizes).
3. Trains the model 3 times, using more data each time. Each time it:
   - splits the data into train / dev / test sets
   - encodes the text as TF-IDF vectors (20,000 features)
   - builds and trains a Keras neural network (see architecture below)
     on the train set
   - evaluates the model on the **dev set**, printing an F1 score and
     saving a ROC curve image
   - evaluates the model on the **test set**, printing an F1 score and
     saving a ROC curve image
4. Saves the final model (from the last, biggest training run).
5. Repeatedly asks for a CSV file to analyze (needs a `review` column).
   For each file, it prints a positive/negative label per review, saves
   the results, and prints one overall verdict for the movie (majority
   vote across all its reviews). It keeps asking for more files until
   you press Enter with no file path.

## Model architecture

The neural network (`mlp_model` in the script) is a Keras `Sequential`
model:

| Layer | Units | Activation | Notes                                                |
|---|---|---|------------------------------------------------------|
| Input / Hidden 1 | 32 | ReLU | `input_dim` = 5,000 (matches the TF-IDF vector size) |
| Dropout | — | — | rate 0.5                                             |
| Hidden 2 | 32 | ReLU |                                                      |
| Dropout | — | — | rate 0.5                                             |
| Output | 1 | Sigmoid | binary positive/negative probability                 |

Compiled with the Adam optimizer and binary cross-entropy loss, trained
with `batch_size=10` and `epochs=10` per cycle. Since TF-IDF produces a
sparse matrix and Keras needs a dense array, the vectorized text is
converted with `.toarray()` before being passed to the model.

**Note:** because the input layer expects exactly 20,000 features,
`max_features=20000` in the `TfidfVectorizer` — changing one requires
changing the other to match.

## Output files

Each training cycle saves two ROC curve images — one for the dev set,
one for the test set:

```
Cycle1_dev_roc.png   Cycle1_test_roc.png
Cycle2_dev_roc.png   Cycle2_test_roc.png
Cycle3_dev_roc.png   Cycle3_test_roc.png
```

The final trained model (Cycle 3) is saved as `model.keras`, and its
matching vectorizer as `vectorizer.pkl`.

## A note on runtime

This Keras model trains noticeably slower than a scikit-learn
`MLPClassifier` — `batch_size=10` means a lot of small update steps per
epoch, and Cycle 3 (39,000 reviews × 10 epochs) will take a while,
especially on CPU. A GPU will speed this up substantially if available.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
python model.py
```



## Reference

Original IMDb: 
Maas et all (2011): Learning word vector for sentiment analysis. Procedings of the 49th Annual Meeting of the Association for Computational Linguistics
Human Technologies. https://aclanthology.org/P11-1015/

MLP Adaptation:
Molankula, B. (2018): Sentiment Analysis of IMDb Dataset. Publisheb on GitHub. https://www.github.com/Balakishan77/Sentiment-Analyis-of-IMDB-dataset/blob/master/imdb_sentiment.py

NOTE: The IMDb dataset was upload directly in the code from "datasets" library (Standforf/imdb).


+ IU-International University of Applied Sciences
+ Course Code: DLBAIPNLP01
+ Tutor: Visieu Lac
+ Author: Gabriel Manu
+ Matriculation ID: 9212512

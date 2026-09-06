## FILE: model.py

## DETAILS:
# Simple movie review sentiment analysis
# Trains the model 3 times on IMDb data: 1,000 / 10,000 / 39,000 samples

### IMPORTING LIBRARIES ###
import pickle
import pandas as pd
import spacy
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

### LOADING AN ENGLISH LANGUAGE CORPUS ###
# Turn off parts of spaCy we don't need, for speed
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

### DATA CLEANING FUNCTION ###
def clean_texts(texts):
    # Cleans many reviews at once (much faster than one at a time)
    # and prints progress so you can see it working
    cleaned = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=100)):
        words = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct]
        cleaned.append(" ".join(words))
        if (i + 1) % 1000 == 0:
            print(f"  cleaned {i + 1}/{len(texts)} reviews")
    return cleaned


### MLP MODEL'S FUNCTION ###
def mlp_model(input_shape):
    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=32, kernel_initializer='uniform', activation='relu', input_dim=input_shape))
    classifier.add(Dropout(0.5))
    # Adding the second hidden layer
    classifier.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.5))
    # Adding the output layer
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

### MODEL'S PERFORMANCE EVALUATION FUNCTION ###
def evaluate_and_plot(model, X, y, name):
    # Runs all the metrics used in this project (F1 score, ROC curve, AUC)
    # on one dataset (dev or test) and saves a plot of the ROC curve.
    probs = model.predict(X, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)

    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, probs)
    print(f"{name}: F1={f1:.2f}  AUC={auc:.2f}")

    fpr, tpr, _ = roc_curve(y, probs)
    plt.plot(fpr, tpr)
    plt.title(name + " ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(name + "_roc.png")
    plt.clf()

### VECTORIZER AND FITTING DATA TO MODEL FUNCTION ###
def train_and_evaluate(data, name):
    print(f"\n{name}: cleaning {len(data)} reviews...")
    data = data.copy()
    data["text"] = data["text"].str.replace("<.*?>", " ", regex=True)  # remove HTML
    data["clean_text"] = clean_texts(data["text"].tolist())

    # Split into train / dev / test
    train, temp = train_test_split(data, test_size=0.3, random_state=42)
    dev, test = train_test_split(temp, test_size=0.5, random_state=42)

    # max_features matches the mlp_model's expected input size (5,000)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train["clean_text"]).toarray()
    X_dev = vectorizer.transform(dev["clean_text"]).toarray()
    X_test = vectorizer.transform(test["clean_text"]).toarray()

    # Calling model's function and model trianing
    model = mlp_model(input_shape=X_train.shape[1])
    model.fit(X_train, train["label"], batch_size=10, epochs=10)

    # Check performance on the dev (validation) set
    print(f"\n{name} - Dev set results:")
    evaluate_and_plot(model, X_dev, dev["label"], f"{name}_dev")

    # Check performance on the test set
    print(f"{name} - Test set results:")
    evaluate_and_plot(model, X_test, test["label"], f"{name}_test")

    return model, vectorizer


### LOADING THE IMDb, COMBINING THE TRAIN AND TEST DATA SETS< AND SH###
print("Loading data...")
data = load_dataset("stanfordnlp/imdb")
df = pd.concat([pd.DataFrame(data["train"]), pd.DataFrame(data["test"])])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

### DATA TRAINING PATITION: FITTING DATASET TO VECTORIZER AND MODEL ###
model, vectorizer = train_and_evaluate(df[0:1000], "Cycle1")
model, vectorizer = train_and_evaluate(df[1000:11000], "Cycle2")
model, vectorizer = train_and_evaluate(df[11000:50000], "Cycle3")

### SAVING MODEL ###
model.save("model.keras")
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

### APPLICATION ON NEW DATA ###
# NOTE: The dataset must have a column called "review" or "reviews".
# Let the user check as many files as they want, one at a time
while True:
    file_path = input("\nCSV file path to analyze (or press Enter to stop): ")

    if not file_path:
        print("Done.")
        break

    try:
        new_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Could not find file: {file_path}")
        continue
    except UnicodeDecodeError:
        try:
            new_df = pd.read_csv(file_path, encoding="latin-1")
        except Exception as e:
            print(f"Could not read file: {e}")
            continue
    except Exception as e:
        print(f"Could not read file: {e}")
        continue

    if "review" not in new_df.columns:
        print(f"Your CSV needs a column named 'review'. Found: {list(new_df.columns)}")
        continue

    new_df["clean_text"] = clean_texts(new_df["review"].tolist())
    X_new = vectorizer.transform(new_df["clean_text"]).toarray()
    probs = model.predict(X_new, verbose=0).flatten()
    new_df["sentiment"] = ["positive" if p > 0.5 else "negative" for p in probs]
    print(new_df[["review", "sentiment"]])

    output_name = file_path.rsplit(".", 1)[0] + "_predictions.csv"
    new_df.to_csv(output_name, index=False)
    print(f"Saved results to {output_name}")

    # Combine all the individual review predictions into one
    # overall sentiment for the movie (majority vote)
    positive_count = (new_df["sentiment"] == "positive").sum()
    negative_count = (new_df["sentiment"] == "negative").sum()
    overall = "POSITIVE" if positive_count >= negative_count else "NEGATIVE"

    print(f"\nOverall movie sentiment: {overall}")
    print(f"({positive_count} positive reviews, {negative_count} negative reviews)")








####################### THE END #######################
# IU-International University of Applied Sciences
# Course Code: DLBAIPNLP01
# Author: Gabriel Manu
# Matriculation ID: 9212512
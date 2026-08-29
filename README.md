# IMDB Review Sentiment Classifier — Simple RNN

A recurrent neural network that classifies free-text movie reviews as positive or negative,
served through a Streamlit app. Built with TensorFlow/Keras.

## Problem

Sentiment classification over natural language, where word *order* carries meaning that a
bag-of-words model throws away ("not good" vs "good"). A recurrent architecture reads the
review token by token and carries state forward, so negation and context survive.

## Data

The Keras built-in **IMDB** dataset — 50,000 labelled movie reviews, pre-tokenised into
integer word indices. The word index is reused at inference time so user-typed text is
encoded exactly the way the model was trained.

## Approach

**Preprocessing** — lowercase and split the review, map each token through the IMDB word
index (offset by 3 for the reserved padding/start/unknown tokens, with unknown words mapped
to index 2), then pad or truncate to a fixed length of 500.

**Model** (`simple_rnn_imdb.h5`):

| Layer | Role |
|---|---|
| `Embedding` | Learns a dense vector per vocabulary token |
| `SimpleRNN` (ReLU) | Reads the sequence and maintains hidden state across timesteps |
| `Dense` (sigmoid) | Outputs P(positive) |

Trained with binary cross-entropy and Adam.

Notebooks: `embedding.ipynb` (how the embedding layer represents words), `simplernn.ipynb`
(architecture and training), `prediction.ipynb` (scoring a single review).

## Running it

```bash
pip install -r requirements.txt
streamlit run main.py
```

Paste a review, press **Classify**, and the app returns the predicted sentiment along with
the raw score (thresholded at 0.5).

## Stack

TensorFlow / Keras · NumPy · Streamlit

## Notes

Built as hands-on practice while following Krish Naik's deep-learning course. A `SimpleRNN`
is deliberately used here rather than an LSTM/GRU — the point of the exercise was to see the
vanilla recurrent cell work, and its limits, before moving to gated architectures.

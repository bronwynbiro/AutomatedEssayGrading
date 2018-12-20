# Automated Essay Grading

## Approach
We explored two distinct models: 1) classical natural language techniques by designing handcrafted features and performing logistic regression and 2) experimented with different Doc2Vec techniques with a LSTM.

## Setup
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
The code can then be run from project.ipynb. 

## Features and Logistic Regression
Features were designed to judge language fluidity, diction, structure, organization, originality and quality of the content. The selected features were as follows.

1.Language quality and originality.

TF-IDF vectors: A TF-IDF vectorizer was trained on the essays and 400 features were selected as unigrams, bigrams, or trigrams. We ensured that each n-gram was observed at least five times in the essay but occurred in no more than 90% of the essays.

Doc2Vec: A Doc2Vec model was built from the essays, and a concatenation of the maximum and minimum vectors for each essay was fed as a feature, as recommended in De Boom et al.(2016). This allows us to encode semantic meaning from the essays, and concatenation performed better than summing or averaging the vectors.

2.Numerical features.

Basic text features: Word count, average word length, and sentence count.
Part of speech counts: Number of nouns, verbs, foreign words, adjectives, adverbs, and conjunctions.
3.Structure and organization.

Punctuation: Number of exclamation marks and question marks.

Then, sklearn was used for k-fold cross validation for the logistic regression. 

## Long Short-Term Memory Network
Long short-term memory units are a modification to recurrent units that use three gates to forget information or preserve it. The model consists of two LSTM layers, a dropout layer, and a dense output layer. The dropout rate was set to 50% to guard against over-fitting.

The first layer of the LSTM has 300 units, 40% of which are dropped for the linear transformation of the input, and 40% of which are dropped for the linear transformation of the recurrent state. The second layer has 64 units, and 40% of units are dropped for the linear transformation of the recurrent state. Then, it runs a Dropout layer, which randomly sets 50% of input units to 0 at each update during training as a way to reduce overfitting. Lastly, it goes to a Dense layer, which is a densely-connected layer. It implements output = activation(dot(input, weights)) where activation is the element-wise ReLu activation function and weights is a weights matrix created by the layer.

## Results
The evaluation metric was Quadratic Weighted Kappa, QWK, as per the ASAP competition. It rates the agreement between two scores and takes into account as the baseline the possibility of agreement occuring by chance, and it typically varies from 0 (random agreement) to 1 (complete agreement). It is also possible to get a negative score if there is less agreeement than expected by chance. 

| Model  |  QWK Score |
|---|---|
|Kaggle competition best score|0.801|
|Logistic regression|0.847|
|Human grading|0.860 |
|Average Doc2Vec and LSTM | 0.907|
|Min+max Doc2Vec and LSTM |0.971
|Sum Doc2Vec and LSTM |0.971|

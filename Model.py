import sys
sys.path.append('/opt/anaconda3/lib/python3.12/site-packages')
sys.path.append('/Users/arjunmukherjee/Library/Python/3.12/lib/python/site-packages')
import numpy as np
import json
import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from collections import Counter

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk import pos_tag, word_tokenize
from sklearn.ensemble import RandomForestClassifier
print(nltk.__file__)

with open("train_data_15.json", "r") as file:
    data = json.load(file)

def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords_list = file.read().splitlines()  # Read and split into a list
    return set(stopwords_list)
stop = load_stopwords("stopwords.txt")

def createDF():
    df_list = []
    for i in range(len(data)):
        player_info = []
        label = data[i]['Label']
        player_info.append(label['Player'])
        player_info.append(label['Skin Tone'])
        player_info.append(label['Teams'])
        player_info.append(label['Year'])
        player_info.append(data[i]['Mention'])
        df_list.append(player_info)
    data_df =  pd.DataFrame(df_list, columns = ['Name', 'Skin Tone', 'Teams', 'Year', 'Mention']) 
    data_df['Label'] = data_df['Skin Tone'].map({'D':1, 'L':0})
    numLight = sum(data_df['Skin Tone'] == 'L')
    #print(sum(data_df['Skin Tone'] == 'L'), sum(data_df['Skin Tone'] == 'D'))
    data_df = data_df.groupby('Skin Tone').apply(pd.DataFrame.sample, n=numLight).reset_index(drop=True)
    #some skin tone is not "d" or "l"
    return data_df


def mostCommonWords():
    limit = 8740
    light = ' '.join(data_df[data_df['Skin Tone'] == 'L']['Mention'].str.lower().values[:limit])
    dark = ' '.join(data_df[data_df['Skin Tone'] == 'D']['Mention'].str.lower().values[:limit])

    def preprocess_and_count(text):
        words = text.split()
        words = [word for word in words if word not in stop]  # Remove stopwords
        return Counter(words)  # Count word frequencies

    # Get word frequency counts
    light_word_counts = preprocess_and_count(light)
    dark_word_counts = preprocess_and_count(dark)

    # Get the most common words (top 20)
    most_common_light = light_word_counts.most_common(20)
    most_common_dark = dark_word_counts.most_common(20)

    return most_common_light, most_common_dark

def getMostWeightedAdj(model, vectorizer, top_n=20):
    """
    Extracts the top N highest-weighted adjectives using basic POS tagging with heuristics.
    
    Parameters:
    - model: Trained logistic regression model.
    - vectorizer: The TF-IDF vectorizer used for transforming text.
    - top_n: Number of top adjectives to return (default is 20).
    
    Returns:
    - List of tuples: (adjective, weight) sorted by weight in descending order.
    """
    # Get feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Tokenize and apply POS tagging (assuming tagging works without downloads)
    tagged_words = pos_tag(feature_names)

    # Filter for adjectives based on POS tags (JJ = adjective, JJR = comparative, JJS = superlative)
    adjectives = [word for word, tag in tagged_words if tag in ['JJ', 'JJR', 'JJS']]

    # Get model coefficients
    coefficients = model.coef_[0]

    # Pair words with their corresponding weights
    word_weights = list(zip(feature_names, coefficients))

    # Filter for adjectives only
    adj_weights = [(word, weight) for word, weight in word_weights if word in adjectives]

    # Sort adjectives by weight in descending order and return the top N
    top_adjectives = sorted(adj_weights, key=lambda x: x[1], reverse=True)[:top_n]
    bottom_adjectives = sorted(adj_weights, key=lambda x: x[1])[:top_n]
    return top_adjectives, bottom_adjectives

def getTopWords(model, vectorizer, X_train, min_freq, top_n):
    """
    Finds the highest and lowest weighted words that appear at least `min_freq` times.

    Parameters:
    - model: Trained logistic regression model.
    - vectorizer: TF-IDF vectorizer used for transforming the text.
    - X_train: Original training text data (before vectorization).
    - min_freq: Minimum frequency a word must appear to be considered (default is 50).
    - top_n: Number of top positive and negative words to return.

    Returns:
    - Tuple of two lists: (top_positive_words, top_negative_words)
    """
    # Get feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Count word frequencies in the original text
    word_counter = Counter(" ".join(X_train).split())

    # Get model coefficients
    coefficients = model.coef_[0]

    # Pair words with their corresponding weights
    word_weights = [(word, weight) for word, weight in zip(feature_names, coefficients) if word_counter[word] >= min_freq]

    # Sort by weight: highest (positive association) and lowest (negative association)
    top_positive_words = sorted(word_weights, key=lambda x: x[1], reverse=True)[:top_n]
    top_negative_words = sorted(word_weights, key=lambda x: x[1])[:top_n]

    return top_positive_words, top_negative_words

def get_racial_bias_keywords(model, vectorizer, keywords):
    """
    Finds the coefficients for specific keywords related to racial bias in the model.

    Parameters:
    - model: Trained logistic regression model.
    - vectorizer: TF-IDF vectorizer used for transforming the text.
    - keywords: List of keywords to check for their coefficients.

    Returns:
    - List of tuples: (keyword, coefficient) sorted by coefficient value.
    """
    # Get feature names (words in the vocabulary)
    feature_names = vectorizer.get_feature_names_out()

    # Get model coefficients (for a binary classification, it's the weight for each word)
    coefficients = model.coef_[0]

    # Pair keywords with their corresponding coefficient
    keyword_coefficients = [(word, coefficients[feature_names.tolist().index(word)] if word in feature_names else None)
                            for word in keywords]

    # Filter out keywords that don't exist in the vocabulary
    keyword_coefficients = [(word, coeff) for word, coeff in keyword_coefficients if coeff is not None]

    # Sort by coefficient value (positive or negative association)
    keyword_coefficients.sort(key=lambda x: x[1], reverse=True)

    return keyword_coefficients


data_df = createDF()
'''from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
counts = count_vect.fit_transform(data_df['Mention'])

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer().fit(counts)

counts_tfidf = transformer.transform(counts)

X_train, X_test, y_train, y_test = train_test_split(counts_tfidf, data_df['Label'], test_size=0.20, random_state=42)
model = LogisticRegression(solver='saga', random_state=42, C=5, penalty='l2',max_iter=1000).fit(X_train, y_train)
prediction = model.predict(X_test)
print("TRAIN: ", model.score(X_train, y_train))
print("TEST: ", model.score(X_test, y_test))
print(classification_report(y_test, prediction))'''

X_train, X_test, y_train, y_test = train_test_split(data_df['Mention'], data_df['Label'], test_size=0.20, random_state=42)
v = TfidfVectorizer(
    stop_words='english',  # Remove common stopwords
    max_features=5000,     # Limit to top 5000 words to prevent overfitting
    ngram_range=(1, 2)     # Use unigrams and bigrams for context
)
train_tf = v.fit_transform(X_train)
test_tf = v.transform(X_test)

model = LogisticRegression(solver='saga', random_state=42, C=5, penalty='l2',max_iter=1000).fit(train_tf, y_train)
#model = RandomForestClassifier(n_estimators=10, min_samples_split=10, max_depth=10, min_samples_leaf=1, max_features='log2').fit(X_train, y_train)

prediction = model.predict(test_tf)
print("TRAIN: ", model.score(train_tf, y_train))
print("TEST: ", model.score(test_tf, y_test))


numzero = (y_test == 1).sum()
numOne = (y_test == 0).sum()
print(numzero, numOne)

'''
feature_names = v.get_feature_names_out()
coefficients = model.coef_[0]
word_weights = list(zip(feature_names, coefficients))
top_positive = sorted(word_weights, key=lambda x: x[1], reverse=True)[:20]
top_negative = sorted(word_weights, key=lambda x: x[1])[:20]
for i in range(20):
    print(i, top_positive[i])

print()
for i in range(20):
    print(i, top_negative[i])
'''

'''keywords = [
    "athletic", "powerful", "natural", "gifted", "quick", "explosive", "fast", "strong", "raw talent", "dynamic",
    "hard-working", "intelligent", "smart", "skilled", "fundamental", "coachable", "focused", "disciplined",
    "determined", "steady", "methodical", "work ethic", "tactical", "speed", "explosiveness", "monster", "beast", "bouncy", "gritty", "hustle"
]

# Get the coefficients for these keywords
bias_keywords = get_racial_bias_keywords(model, v, keywords)

# Print the coefficients and sort them by strength of association
print("Keywords and their coefficients:")
for word, coeff in bias_keywords:
    print(f"{word}: {coeff:.4f}")



topWords, bottomWords = getTopWords(model, v, X_train, 75, 20)

print("Top 20 Highest Weighted Words (Black):")
for word, weight in topWords:
    print(f"{word}: {weight:.4f}")
print("Top 20 lowest words(White):")
for word, weight in bottomWords:
    print(f"{word}: {weight:.4f}")
#train_tf, test_tf = convertTDIF(data_df)'''

#NEED TO MAKE SAME AMOUNT OF WHITE MENTIONS AS BLACK MENTIONS IN ALL DATASETS



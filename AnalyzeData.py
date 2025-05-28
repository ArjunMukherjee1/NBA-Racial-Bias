import sys
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
    light_word_counts = preprocess_and_count(light)
    dark_word_counts = preprocess_and_count(dark)
    most_common_light = light_word_counts.most_common(20)
    most_common_dark = dark_word_counts.most_common(20)
    return most_common_light, most_common_dark

def getMostWeightedAdj(model, vectorizer, top_n=20):
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
    feature_names = vectorizer.get_feature_names_out()
    word_counter = Counter(" ".join(X_train).split())
    coefficients = model.coef_[0]
    word_weights = [(word, weight) for word, weight in zip(feature_names, coefficients) if word_counter[word] >= min_freq]
    top_positive_words = sorted(word_weights, key=lambda x: x[1], reverse=True)[:top_n]
    top_negative_words = sorted(word_weights, key=lambda x: x[1])[:top_n]
    return top_positive_words, top_negative_words

def get_racial_bias_keywords(model, vectorizer, keywords):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    keyword_coefficients = [(word, coefficients[feature_names.tolist().index(word)] if word in feature_names else None)
                            for word in keywords]
    keyword_coefficients = [(word, coeff) for word, coeff in keyword_coefficients if coeff is not None]
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
    #max_features=5000,     # Limit to top 5000 words to prevent overfitting
    ngram_range=(1, 2)     # Use unigrams and bigrams for context
)
train_tf = v.fit_transform(X_train)
test_tf = v.transform(X_test)

model = RandomForestClassifier(n_estimators=200, min_samples_split=2, max_depth=20, min_samples_leaf=1, max_features='log2', class_weight='balanced').fit(train_tf, y_train)
# training the Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
model = LogisticRegression(solver='saga', random_state=42, C=5, penalty='l2',max_iter=1000).fit(train_tf, y_train)
#model = RandomForestClassifier(n_estimators=200, min_samples_split=2, max_depth=20, min_samples_leaf=1, max_features='log2', class_weight='balanced').fit(train_tf, y_train)
#model = XGBClassifier(subsample=0.5, max_depth=24).fit(train_tf, y_train) #super long time
#model = GradientBoostingClassifier(n_estimators=200, min_samples_split=2, max_depth=20, min_samples_leaf=1, max_features='log2').fit(train_tf, y_train)

#model = DecisionTreeClassifier(criterion='entropy', max_depth=150, min_impurity_decrease=0.000007).fit(train_tf, y_train) #improve

#model = AdaBoostClassifier(n_estimators=200, random_state=42).fit(train_tf, y_train) #long time

#model = MultinomialNB().fit(X_train, y_train)

prediction = model.predict(test_tf)
print("TRAIN: ", model.score(train_tf, y_train))
print("TEST: ", model.score(test_tf, y_test))
print(classification_report(y_test, prediction))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, train_tf, y_train, cv=5)

print("Cross-validation scores:", scores)
print(scores.mean())
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


keywords = [
    "athletic", "powerful", "natural", "gifted", "quick", "explosive", "fast", "strong", "raw talent",
    "hard-working", "intelligent", "smart", "skilled", "fundamental", "focused", "disciplined",
    "determined", "steady", "methodical", "work ethic", "tactical", "speed", "explosiveness", "monster", "beast", "bouncy", "gritty", "hustle", "lazy", "powerhouse", "humble",
    "scrappy"
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


'''from transformers import BertTokenizer
X_train_list = X_train.astype(str).tolist()
X_test_list = X_test.astype(str).tolist()
new_y_train = y_train.reset_index(drop=True)
new_y_test = y_test.reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize text data
train_encodings = tokenizer(X_train_list, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test_list, truncation=True, padding=True, max_length=128)
print()
import torch

class CommentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        print("Dataset length:", len(self.labels))
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create dataset objects
train_dataset = CommentDataset(train_encodings, new_y_train)
test_dataset = CommentDataset(test_encodings, new_y_test)

from transformers import BertForSequenceClassification

import os
MODEL_PATH = "./saved_model"

if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
else:
    print("No saved model found. Initializing a new model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, ignore_mismatched_sizes = True)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',      # Save model output here
    num_train_epochs=3,          # Number of training epochs
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,  
    warmup_steps=500,            # Learning rate warmup
    weight_decay=0.01,           # Regularization
    logging_dir='./logs',        # Logging directory
    evaluation_strategy="epoch", # Evaluate at the end of each epoch
)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
#trainer.train()

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Train and save the model
if not os.path.exists(MODEL_PATH):
    print("Training model...")
    trainer.train()
    print("Saving trained model...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

# Evaluate model
results = trainer.evaluate()

print("Evaluation Results:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")  # Formatting to 4 decimal places
input_lengths = [len(tokenizer.tokenize(text)) for text in X_train_list]
print(f"Avg token length: {np.mean(input_lengths)}, Max token length: {np.max(input_lengths)}")'''


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']  # Compound score: -1 (most negative) to +1 (most positive)
data_df['Sentiment'] = data_df['Mention'].apply(get_sentiment)

#Compute average sentiment by race
sentiment_summary = data_df.groupby('Skin Tone')['Sentiment'].mean().reset_index()
print(sentiment_summary)
#print(2*(sentiment_summary['Sentiment'][1] - sentiment_summary['Sentiment'][0])/(sentiment_summary['Sentiment'][1] + sentiment_summary['Sentiment'][0]))
from scipy import stats
dark_skin_sentiment = data_df[data_df['Skin Tone'] == 'D']['Sentiment']
light_skin_sentiment = data_df[data_df['Skin Tone'] == 'L']['Sentiment']
t_stat, p_value = stats.ttest_ind(dark_skin_sentiment, light_skin_sentiment)
print(f"T-statistic: {t_stat}, P-value: {p_value}")

#histogram
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data=data_df, x='Sentiment', hue='Skin Tone', kde=True, bins=20, multiple="dodge")
plt.title("Sentiment Distribution by Player Race")
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.legend(title="Skin Tone", labels=["Lighter", "Darker"])
plt.show()


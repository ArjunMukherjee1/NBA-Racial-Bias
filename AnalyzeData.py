import numpy as np
import json
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize


from nltk.corpus import stopwords
from collections import Counter

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk import pos_tag, word_tokenize
from sklearn.ensemble import RandomForestClassifier


with open("train_data_15.json", "r") as file:
    data = json.load(file)
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords_list = file.read().splitlines()  # Read and split into a list
    return set(stopwords_list)
stop = load_stopwords("stopwords.txt")

#Creates a balanced Dataframe from the loaded JSON data
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

    #Darker-skiinned players labeled as 1, lighter-skinned as 0
    data_df['Label'] = data_df['Skin Tone'].map({'D':1, 'L':0})
    numLight = sum(data_df['Skin Tone'] == 'L')
    #print(sum(data_df['Skin Tone'] == 'L'), sum(data_df['Skin Tone'] == 'D'))

    #Removes mentions of darker-skinned players to make the dataset balanced
    data_df = data_df.groupby('Skin Tone').apply(pd.DataFrame.sample, n=numLight).reset_index(drop=True)
    return data_df

#identifies most common words in the mentions of both groups
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

#Finds highest weighted adjectives
def getMostWeightedAdj(model, vectorizer, top_n=20):
    feature_names = vectorizer.get_feature_names_out()
    tagged_words = pos_tag(feature_names)

    #Filters for adjectives based on tags (JJ = adjective, JJR = comparative, JJS = superlative)
    adjectives = [word for word, tag in tagged_words if tag in ['JJ', 'JJR', 'JJS']]
    
    coefficients = model.coef_[0]
    word_weights = list(zip(feature_names, coefficients))

    #Gets a list of only adjectives
    adj_weights = [(word, weight) for word, weight in word_weights if word in adjectives]
    top_adjectives = sorted(adj_weights, key=lambda x: x[1], reverse=True)[:top_n]
    bottom_adjectives = sorted(adj_weights, key=lambda x: x[1])[:top_n]
    return top_adjectives, bottom_adjectives

#Finds the highest weighted words both positive and negative
def getTopWords(model, vectorizer, X_train, min_freq, top_n):
    feature_names = vectorizer.get_feature_names_out()
    word_counter = Counter(" ".join(X_train).split())
    coefficients = model.coef_[0]
    word_weights = [(word, weight) for word, weight in zip(feature_names, coefficients) if word_counter[word] >= min_freq]
    top_positive_words = sorted(word_weights, key=lambda x: x[1], reverse=True)[:top_n]
    top_negative_words = sorted(word_weights, key=lambda x: x[1])[:top_n]
    return top_positive_words, top_negative_words

#Returns sorted list of words and their corresponding weights for every word in the list "keywords"
def get_racial_bias_keywords(model, vectorizer, keywords):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    keyword_coefficients = [(word, coefficients[feature_names.tolist().index(word)] if word in feature_names else None)
                            for word in keywords]
    keyword_coefficients = [(word, coeff) for word, coeff in keyword_coefficients if coeff is not None]
    keyword_coefficients.sort(key=lambda x: x[1], reverse=True)
    return keyword_coefficients

#Create dataframe
data_df = createDF()

#Separate data for training and testing
X_train, X_test, y_train, y_test = train_test_split(data_df['Mention'], data_df['Label'], test_size=0.20, random_state=42)

#Convert mentions to a matrix of TF-IDF 
v = TfidfVectorizer(
    stop_words='english',  
    #max_features=5000,
    ngram_range=(1, 2)     #Use terms of 1 or 2 words
)
train_tf = v.fit_transform(X_train)
test_tf = v.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#Train models
model = LogisticRegression(solver='saga', random_state=42, C=5, penalty='l2',max_iter=1000).fit(train_tf, y_train) #Top performance

#model = RandomForestClassifier(n_estimators=200, min_samples_split=2, max_depth=20, min_samples_leaf=1, max_features='log2', class_weight='balanced').fit(train_tf, y_train) #2nd best performance

#model = XGBClassifier(subsample=0.5, max_depth=24).fit(train_tf, y_train) #super long time

#model = GradientBoostingClassifier(n_estimators=200, min_samples_split=2, max_depth=20, min_samples_leaf=1, max_features='log2').fit(train_tf, y_train)

#model = DecisionTreeClassifier(criterion='entropy', max_depth=150, min_impurity_decrease=0.000007).fit(train_tf, y_train) #improve

#model = AdaBoostClassifier(n_estimators=200, random_state=42).fit(train_tf, y_train) #long time

#model = MultinomialNB().fit(X_train, y_train)

#Test model
prediction = model.predict(test_tf)
print("TRAIN: ", model.score(train_tf, y_train))
print("TEST: ", model.score(test_tf, y_test))
print(classification_report(y_test, prediction))

from sklearn.model_selection import cross_val_score
#Cross validation scores
#scores = cross_val_score(model, train_tf, y_train, cv=5)

#print("Cross-validation scores:", scores)
#print(scores.mean())


#top_adjs, bottom_adjs = getMostWeightedAdj(model, v, top_n=20)


#print("Top weighted adjectives for darker-skinned players:")
#for word, weight in top_adjs:
#    print(f"{word}: {weight:.3f}")

#Confusion Matrix
'''from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, prediction, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Lighter','Darker'])

fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax)
plt.title("Confusion Matrix")
plt.show()
'''

feature_names = v.get_feature_names_out()
coefficients = model.coef_[0]
word_weights = list(zip(feature_names, coefficients))
#top_positive = sorted(word_weights, key=lambda x: x[1], reverse=True)[:250]
#top_negative = sorted(word_weights, key=lambda x: x[1])[:250]
#for i in range(250):
    #print(i, top_positive[i])

#print()
#for i in range(250):
    #print(i, top_negative[i])

#Racial bias keywords
keywords = [
    #Work Ethic and Character
    "hardworking", "scrappy", "gritty", "hustle", 
    "grinder", "overachiever", "hard-working", 
    "work ethic", "determined", "focused", 
    "steady", "hustling",

    #Intelligence
    "smart", "cerebral", "high IQ", "basketball IQ", "coachable", 
    "heady", "instinctual", "disciplined", "savvy", "methodical", 
    "polished", "intelligent", "tactical", "poised", "strategic", 
    "aware", "intuitive", "IQ",

    #Athleticism
    "athletic", "freakish", "bouncy", "strong", 
    "fast", "raw talent", "gifted", "quick", 
    "powerful", "natural", "speed", "explosiveness", 
    "monster", "agile", "agility", "springy", 
    "high flyer", "muscular", "muscle", "powerhouse", 
    "long", "machine",

    #Temperment
    "aggressive", "emotional", "hot-headed", "fiery", 
    "passionate", "intense", "volatile", "lazy", "bully",

    #Leadership
    "leader", "captain", "vocal", "mature", "professional", 
    "role model", "humble", "grounded", "arrogant", "soldier",

    #Skill or Creativity
    "flashy", "finesse", "creative", "unpredictable", "streetball", 
    "fundamental", "textbook", "wild", "skilled",

    #Other
    "thug", "urban", "articulate", 
    "intimidating", "selfish", "greedy", "me-first", 
    "unfortunate", "afraid", "trouble", "smart play", 
    "entertaining", "tough", "slow", "weak", 
    "perserve", "unbelievable", "spectacular", 
    "unstoppable", "beauty", "unlucky"
]
bias_keywords = get_racial_bias_keywords(model, v, keywords)

print("Keywords and their coefficients:")
for word, coeff in bias_keywords:
    print(f"{word}: {coeff:.4f}")

'''

topWords, bottomWords = getTopWords(model, v, X_train, 75, 20)

print("Top 20 Highest Weighted Words (Black):")
for word, weight in topWords:
    print(f"{word}: {weight:.4f}")
print("Top 20 lowest words(White):")
for word, weight in bottomWords:
    print(f"{word}: {weight:.4f}")
#train_tf, test_tf = convertTDIF(data_df)'''


'''from transformers import BertTokenizer
#Converts mentions into lists of strings and resets indices of the skin color label to match
X_train_list = X_train.astype(str).tolist()
X_test_list = X_test.astype(str).tolist()
new_y_train = y_train.reset_index(drop=True)
new_y_test = y_test.reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Tokenize text data
train_encodings = tokenizer(X_train_list, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test_list, truncation=True, padding=True, max_length=128)
import torch

class CommentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        #print("Dataset length:", len(self.labels))
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
MODEL_PATH = "./results"

if os.path.exists(MODEL_PATH):
    print("Loading saved model")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
else:
    print("No saved model found. Initializing a new model")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, ignore_mismatched_sizes = True)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',      
    num_train_epochs=3,
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,  
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
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


trainer.train()

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
    return scores['compound']  # Compound score: ranges from -1 to 1
data_df['Sentiment'] = data_df['Mention'].apply(get_sentiment)

#Compute average sentiment by race
sentiment_summary = data_df.groupby('Skin Tone')['Sentiment'].mean().reset_index()
print(sentiment_summary)
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




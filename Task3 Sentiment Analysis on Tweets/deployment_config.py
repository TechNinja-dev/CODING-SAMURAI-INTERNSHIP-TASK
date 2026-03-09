import numpy as np
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# check wordnet
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib,os

class TweetAnalysis:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.model=joblib.load("sentiment.pkl")
    
    def clean_text(self,text):

        text = text.lower()

        text = re.sub(r"http\S+", "", text)   # remove links
        text = re.sub(r"@\w+", "", text)      # remove mentions
        text = re.sub(r"#\w+", "", text)      # remove hashtags
        text = re.sub(r"[0-9]", "", text)     # remove numbers
        text = re.sub(r"[^\w\s]", "", text)   # remove punctuation

        text = text.strip()

        return text

    def preprocess(self,text):
        text=self.clean_text(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        words = text.split()

        words = [word for word in words if word not in self.stop_words]

        words = [self.lemmatizer.lemmatize(word) for word in words]

        return " ".join(words)
    
    def predict(self,text):
        
        tweet=self.preprocess(text)
        rel=self.model.predict(tweet)[0]
        return rel
    

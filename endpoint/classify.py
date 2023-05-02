from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core import Run
import json
import os
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import pickle
from azureml.core import Run


nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(
        stopwords.words('english'))]
    text = ' '.join(text)
    return text


def init():
    global model, vectorizer
    with open(os.path.join(
            os.getenv('AZUREML_MODEL_DIR'), 'sentimentModel.pkl'), 'rb') as f:
        model = pickle.load(f)
        vectorizer = pickle.load(f)


def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    text = pd.DataFrame(data, columns=['text'])
    text = text['text'].apply(preprocess)
    text = vectorizer.transform(text).toarray()
    y_hat = model.predict(text)
    return y_hat.tolist()

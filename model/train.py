import os
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import nltk
import re
import pandas as pd
from azureml.core import Run


nltk.download('stopwords')
nltk.download('wordnet')

# get the Azure ML run object (Azure Model Functions)
run = Run.get_context()

data = pd.read_csv('sample.csv',
                   encoding='latin-1', header=None)
data = data.sample(100000)
text, sentiment = data[5], data[0]


lemmatizer = WordNetLemmatizer()


def preprocess(text):
    global lemmatizer
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(
        stopwords.words('english'))]
    text = ' '.join(text)
    return text


text = text.apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    text, sentiment, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

model = LogisticRegression()
model.fit(X_train, y_train)

# log the accuracy of the model
run.log("Accuracy", accuracy_score(y_test, model.predict(X_test)))

# get existing path where model will be saved (standard Azure path)
model_path = 'outputs/sentimentModel.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
    pickle.dump(vectorizer, f)

# one must upload the model explicitly to ge the permission for model registry
run.upload_file(name=model_path, path_or_stream=model_path)
# register the model
run.register_model(model_name='sentiment_model', model_path=model_path)

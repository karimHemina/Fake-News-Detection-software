import pickle

import nltk
import streamlit
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re


def stemming(content):
    porter_stemmer = PorterStemmer()
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [porter_stemmer.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content


def fake_news_detection(news):
    model = pickle.load(open("models/fake_news_logistic_regression_model.sav", 'rb'))
    vectorizer = pickle.load(open("models/fake_news_logistic_regression_tfidf_vectorizer.sav", 'rb'))

    prediction = model.predict(
        vectorizer.transform([stemming(news)])
    )

    if (prediction[0] == 0):
        return "this is Real"
    else:
        return "This is fake"

def main():
    nltk.download('stopwords')
    streamlit.title("Fake News Detection")

    news = streamlit.text_input('News')

    prediction = ''
    if streamlit.button("News Classification"):
        prediction = fake_news_detection(news)

    streamlit.success(prediction)

main()

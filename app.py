import pickle
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('dancing')
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = pickle.load(open('LR_text_classification_model.pkl', 'rb'))


def main():
    st.title('Biomedical Cancer Report Classification')
    # input vriable
    Text = st.text_input('Enter Medical Report Text')

    if st.button("Predict type of cancer"):
        df = pd.DataFrame({'Text': [Text]})

        # tokenize text
        def transform_text(text):
            text = text.lower()
            text = nltk.word_tokenize(text)

            y = []
            for i in text:
                if i.isalnum():
                    y.append(i)

            text = y[:]
            y.clear()

            for i in text:
                if i not in stopwords.words('english') and i not in string.punctuation:
                    y.append(i)

            text = y[:]
            y.clear()

            for i in text:
                y.append(ps.stem(i))
            return " ".join(y)

        df['transformed_text'] = df['Text'].apply(transform_text)

        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(df['transformed_text']).toarray()

        prediction = model.predict(X)
        ans = ("Your type of cancer according to our model is " , prediction[0])
        st.header(ans)


if __name__ == '__main__':
    main()
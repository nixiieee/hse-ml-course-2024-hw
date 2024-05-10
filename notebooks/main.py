import click
import pandas as pd
import numpy as np
import nltk
import os
import sys
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pandas.api.types import is_object_dtype, is_numeric_dtype
 
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stem_delete_stopwords(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    text = ' '.join(map(stemmer.stem, text.split(' ')))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_text)

def is_float(element: any) -> bool:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def train_model(data, test, split, model):
    if os.path.exists('../data/' + data):
        df = pd.read_csv('../data/' + data)
        check_data(df)
    else:
        raise OSError("Incorrect path to data file")
    df = preprocess(df)
    if test:
        if os.path.exists('../data/' + test):
            df_test = pd.read_csv('../data/' + test)
            check_data(df_test)
        else:
            raise OSError("Incorrect path to data file")
        df_test = preprocess(df_test)
    else:
        if is_float(split):
            split = float(split)
            if 0 < split and split < 1:
                df, df_test = train_test_split(df, test_size=split, random_state=42)
            else:
                raise ValueError("Incorrect split value: should be between 0 and 1")
        else:
            raise TypeError("Incorrect type of split variable")
            
    x_train = df['text']
    y_train = df['rating']
    x_test = df_test['text']
    y_test = df_test['rating']

    pipeline = Pipeline(
            [
                ("vectorizer", TfidfVectorizer()),
                ("model", LogisticRegression(random_state=42, C=10, penalty='l2', solver='lbfgs')),
            ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    print(f1_score(y_pred, y_test))

    if model:
        with open(model, 'wb') as file:
            pickle.dump(pipeline, file)
    return


def predict_model(data, path_to_model):
    if data.endswith('.csv'):
        if os.path.exists('../data/' + data):
            df = preprocess(pd.read_csv('../data/' + data))
            X = df['text']
        else:
            raise OSError("Incorrect path to data file")
    else:
        X = [stem_delete_stopwords(data)]
    with open(path_to_model, 'rb') as file:
        pipeline = pickle.load(file)
    y_pred = pipeline.predict(X)
    for i in y_pred:
        print(i)
    return y_pred


def preprocess(df):
    df['text'] = df['title'] + ' ' + df['text']
    df = df.dropna()
    df.loc[:,'text'] = df['text'].apply(stem_delete_stopwords)
    df = df.drop(columns=['type', 'published_date', 'published_platform', 'helpful_votes', 'title'])
    df['rating'] = (df['rating'] > 3) # будем считать негативными отзывами те, у которых рейтинг <= 3
    return df


correct_columns = ['helpful_votes',
        'published_date',
        'published_platform',
        'rating',
        'text',
        'title',
        'type']


def check_data(df):
    if 'Unnamed: 0' in list(df.columns):
        df = df.drop(columns=['Unnamed: 0'])
    if sorted(list(df.columns)) != correct_columns:
        raise TypeError(f"Incorrect data file format: {sorted(list(df.columns))}")
    if not(is_object_dtype(df['text']) and is_object_dtype(df['title']) and is_numeric_dtype(df['rating'])):
        raise TypeError("Incorrect data file format: column type mismatch")
    if df.empty:
        raise TypeError("Incorrect data file format: data is empty")
    return
    

@click.command()
@click.option('--data', required=True, help='Path to the dataset')
@click.option('--test', required=False, help='Test dataset')
@click.option('--split', required=False, type=float,  help='Proportion of test in the provided dataset')
@click.option('--model', required=True, help='Path to the model')
@click.argument('command')
def main(command, data, test, split, model):
    if command == 'train':
        train_model(data, test, split, model)
    elif command == 'predict':
        predict_model(data, model)
    else:
        raise NotImplementedError("Unknown command")


if __name__ == '__main__':
    main()

# import libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer


def load_data(database_filepath):

    """
    Load dataset from database

    Arguments:
        Path to databse where the dataframe is stored
    Output:
        X --> dataframe containing features
        y --> dataframe containing labels
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)

    #Remove column as it has only null values
    df = df.drop('child_alone',axis=1)

    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns  # This will be used for visualization purpose
    return X,y, category_names


def tokenize(text):
    """
    Tokenize function

    Arguments:
        text -> Relevant text that needs to be tokenized
    Output:
        clean_tokens -> List of lemmatized tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # detect all url in order to replace them
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "url_placeholder")

    # create tokens and initialize lemmatizer
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # create an array with lemmatized tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class

    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)



def build_model():
    """
    Build a machine learning pipeline

    Output:
    A machine learning pipeline for text messages
    """

    #define pipeline with feature union and gridsearch parameters
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer',
                 CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer',
             StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters_grid = {

        'clf__estimator__n_estimators': [50, 200],
        'clf__estimator__min_samples_split': [2, 4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters_grid, n_jobs=-1)
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates a machine learning model in
    terms of performance

    Arguments:
        model -> Machine learning model
        X_test --> Test data for features
        Y_test --> Test data for labels
        category_names --> category names

    Output:
        performance metrics -> Performance of the
        machine learning model
    """
    # predict y values for the test set
    y_prediction_test = model.predict(X_test)

    # calculate and print the accuracy of the model
    accuracy = (y_prediction_test == Y_test).mean().mean()
    print('The accuracy of the model is {0:.2f}%'.format(accuracy*100))

    #print the classification report
    print(classification_report(np.hstack(Y_test.values), np.hstack(y_prediction_test),target_names=category_names ))

def save_model(model, model_filepath):
    """
    This function saves the model to model filepath

    Arguments:
        model -> Machine learning model
        model_filepath --> location where the model is stored
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
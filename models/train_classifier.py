# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger'])


# import libraries
import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin



class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    # fetching data from SQL
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    # splitting DataFrame info inputs (X), and outputs (Y)
    X = df[['message','original','genre']]
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    # function to tokenize given message string
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    # defining the ML pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # defining parameters for GridSearch
    parameters ={
        'clf__estimator__n_estimators': [i for i in range(50, 200, 50)],
        'clf__estimator__max_depth': [2, 5, 10],
        'clf__estimator__max_features': ['sqrt', 'log2']
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    #function to print out F1-score, Precision, and Recall
    Y_pred = model.predict(X_test)
    f1_array=[]
    precision_array=[]
    recall_array=[]
    result = pd.DataFrame(index = Y_test.columns, columns = ['f1-score','precision','recall'])
    k = 0
    # F1-score, Precision, and Recall are calculated individually for each category, saved, then printed
    for column in Y_test:   
        f1_array.append(f1_score(Y_test[column], Y_pred[:,k],average='macro'))
        precision_array.append(precision_score(Y_test[column], Y_pred[:,k],average='macro'))
        recall_array.append(recall_score(Y_test[column], Y_pred[:,k],average='macro'))
        k+=1
    result['f1-score'] = f1_array
    result['precision'] = precision_array
    result['recall'] = recall_array
    print(result)


def save_model(model, model_filepath):
    # exports the model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        # The data is loaded, then split into train and test datasets. Only the message column of X is used to train the model.
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X.message, Y, test_size=0.2)
        
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
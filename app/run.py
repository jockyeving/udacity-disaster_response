import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///' + '../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    aid_counts = df.groupby('aid_related').count()['message']
    aid_names = list(aid_counts.index)
    
    weather_counts = df.groupby('weather_related').count()['message']
    weather_names = list(weather_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    graphs2 = [
        {
            'data': [
                Bar(
                    x = aid_names,
                    y = aid_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Based on Aid Request',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Is aid being requested?"
                }
            }
        }
    ]
    graphs3 = [
        {
            'data': [
                Bar(
                    x = weather_names,
                    y = weather_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Based on Being Related to Weather',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Is the message weather related?"
                }
            }
        }
    ]
    
  
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    ids2 = ["graph-{}".format(i) for i, _ in enumerate(graphs2)]
    graphJSON2 = json.dumps(graphs2, cls=plotly.utils.PlotlyJSONEncoder)
    ids3 = ["graph-{}".format(i) for i, _ in enumerate(graphs3)]
    graphJSON3 = json.dumps(graphs3, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, ids2=ids2, ids3=ids3, graphJSON=graphJSON, graphJSON2=graphJSON2, graphJSON3=graphJSON3)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
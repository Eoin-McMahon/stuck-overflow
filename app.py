from flask import Flask , render_template, request
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

app = Flask(__name__)

tagGet = pd.read_csv("cleantitletag.csv", encoding='ISO-8859-1')
ansGet = pd.read_csv("cleanbodytag.csv", encoding='ISO-8859-1')

tagModel = load_model("qTot.hdf5")

def getAnswer(tag):
    text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf_svm', SGDClassifier())])
    model = joblib.load("joblib.h5")
    x = str(model.predict([tag]))
    return x[2:len(x)-3]

def getKey(question):
    return tagModel.predict(question)

@app.route('/')
def hello_world():
    return render_template('./index.html', text="")

if __name__ == 'main':
    app.run()
    
@app.route("/search/", methods=["POST", "GET"])
def search():
    text = request.form['question']
    processed_text = getAnswer(text)
    return render_template('./index.html', text=processed_text)

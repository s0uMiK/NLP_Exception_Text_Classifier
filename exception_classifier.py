from flask import Flask,render_template,session,url_for,redirect,request
import numpy as np 
import os
import werkzeug
# from tensorflow.keras.models import load_model
from nltk.stem import PorterStemmer
from joblib import load
import pickle
import os ,os.path
from pandas import DataFrame
import pandas as pd
import dask.dataframe as dd
import csv
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField,RadioField
from flask_wtf.file import FileField, FileRequired, FileAllowed

# ================================================================================================
import spacy

import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import en_core_web_sm

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = en_core_web_sm.load()
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

from sklearn.base import TransformerMixin

def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

# ====================================================================================
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import requests
from tensorflow.keras.models import load_model

tokenizer = Tokenizer()

def decode_sentiment(score):
    return "Business Exception" if score>0.5 else "System Exception"

def lstmPredict(model,exceptions):
	tokenizer.fit_on_texts(exceptions)
	processed = pad_sequences(tokenizer.texts_to_sequences(exceptions),maxlen =30)
	print(type(processed))
	op = model.predict(processed)
	processed = [decode_sentiment(score) for score in op]
	# print(processed)
	lines = list()
	for i in range(len(processed)):
		line = list()
		line.append(processed[i])
		line.append(exceptions[i])
		s=" : "
		s= s.join(line)
		exceptions[i] = s


	# data = json.dumps({"signature_name": "serving_default", "instances": processed.tolist()})
	# headers = {"content-type": "application/json"}
	# json_response = requests.post('http://localhost:5050/v1/models/lstm_model:predict', data=data, headers=headers)
	# print(json_response)
	# predictions = list(json.loads(json_response.text)['predictions'])

model_file_name = 'model.pkl' 
def piplinePredict(model,exceptions):
	ops = model.predict(exceptions)
	lines = list()
	for i in range(len(ops)):
		line = list()
		line.append(ops[i])
		line.append(exceptions[i])
		s=" : "
		s= s.join(line)
		exceptions[i] = s


def stemming(wds):
	porter = PorterStemmer()
	for i in range(len(wds)):
		wds[i] = porter.stem(wds[i])

key_bus = ['contain','error','fields','mandatory','missing','found','template','file','field', 'input','processed','mail','invite','manually','id']
key_sys = ['internal','exception','memory','expression','code','stage','site','perform','evaluate','cleanup','time','hresult','failed','operation','unable','issue','data','fetch','unexpected','automatically','set','launch']

porter = PorterStemmer()
def brute_force_audit(txts):
	stemming(key_bus)
	stemming(key_sys)
	for i in range(len(txts)):
		sent = txts[i].lower()
		txts[i] = txts[i].split()
		words = sent.split()
		stemming(words)
		Bis_check =  any(item in key_bus for item in words)
		Sys_check =  any(item in key_sys for item in words)
		if Bis_check == True:
			op = ['Business Exception : ']
			for j in txts[i]:
				op.append(j)
			s=" "
			s = s.join(op)
			txts[i] = s
		elif Sys_check == True:
			op = ['System Exception : ']
			for j in txts[i]:
				op.append(j)
			s=" "
			s = s.join(op)
			txts[i] = s


app = Flask(__name__)
app.config['SECRET_KEY'] = 'nlp_classifier'
# f = open('text_classification.pkl','rb')


@app.route("/",methods = ['GET','POST'] )
def index():
	
	dat = list()
	op=list()
	if request.method =='POST':
		f = request.files['csvfile']
		cf = isinstance(f,werkzeug.datastructures.FileStorage)
		if f.filename!='':
			f.save(os.path.join(os.getcwd(),f.filename))
			fname = str(f.filename)
			if fname !="":
				df = dd.read_csv(fname)
				df = df.iloc[:,0].to_frame()
				df.to_csv('real_input.csv')

				DIR = str(os.getcwd())+"/real_input.csv"
				num =  (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
				for i in range(num):
  		 			df1 = pd.read_csv(DIR+'/'+str(i)+'.part')
  		 			dat = dat + list(df1.iloc[:,1])

				ana_type = request.form.getlist('Class')
				if '3' in ana_type:
					brute_force_audit(dat)
					opdf = DataFrame(dat,columns=['Brute Force Results'])
					opdf.to_csv('brute_force_audit_results.csv')
				elif '2' in ana_type:
					model = load(model_file_name)
					piplinePredict(model,dat)
					op_pip_df = DataFrame(dat,columns=['Spacy Classifier Results'])
					op_pip_df.to_csv('spacy_audit_results.csv')
				elif '1' in ana_type:
					lstm_model = load_model("lstm_model.h5")
					lstmPredict(lstm_model,dat)
					op_lstm_df = DataFrame(dat,columns=['LSTM Classifier Results'])
					op_lstm_df.to_csv('lstm_audit_results.csv')

					
					
			else:
				dat.clear()
		else:
			dat.clear()
		
		# check for type of analysis
		# pass the info in dat to that prediction function


	
	return render_template('class_home.html',data = dat)	


if __name__== '__main__':
	
	app.run()

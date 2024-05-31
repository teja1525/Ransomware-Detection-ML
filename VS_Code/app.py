import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import sqlite3

import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
import pandas as pd
import numpy as np
import pickle
import sqlite3
import random
import smtplib 
from email.message import EmailMessage
from datetime import datetime

warnings.filterwarnings('ignore')



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/home')
def home():
	return render_template('home.html')


@app.route("/notebook")
def notebook1():
    return render_template("Notebook.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features= [float(x) for x in request.form.values()]
    print(int_features,len(int_features))
    final4=[np.array(int_features)]
    model = joblib.load('model.sav')
    predict = model.predict(final4)

    if predict == 0:
        output='Benign!'

    elif predict == 1:
        output = 'Ransomware Detected!'
    

    return render_template('prediction.html', output=output)



if __name__ == "__main__":
    app.run(debug=True)

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 09:46:00 2021

@author: LENOVO
"""


from flask import Flask, render_template,url_for,request
import pandas as pd
import pickle

loaded_model= pickle.load(open('random_forest.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    df = pd.read_csv("real_2016.csv")
    df.drop(["PM 2.5"], axis=1, inplace=True)
    my_predict = loaded_model.predict(df.values)
    my_predict = my_predict.tolist()
    return render_template('result.html', prediction = my_predict)


if __name__ == "__main__":
    app.run(debug=True)
 

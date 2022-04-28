# Import necessary modules
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import sklearn
import joblib


app = Flask(__name__)
model = joblib.load(open('rf_model_final.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html')

@app.route("/predict",methods=['GET','POST'] )
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output=["Fraud" if prediction[0]==1 else "Not Fraud"][0]
    return render_template('index2.html', prediction_text='The transaction is {}'.format(output), prediction=prediction,final_features=final_features)

# @app.route('/results',methods=['POST'])
# def results():
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#     output=["Fraud" if prediction[0]==1 else "Not Fraud"]
#     return jsonify(output)

if __name__ == "__main__":
    #columns = joblib.load(open("columns","rb"))
    app.run(debug=True, host="0.0.0.0")





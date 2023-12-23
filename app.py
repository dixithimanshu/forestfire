# Imports

from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

## Below command will install all the libraries from below file:
## pip install -r requirements.txt

# For running the 'app.py' file in terminal use command 'python app.py'

app = Flask(__name__)


# Ridge Regressor Model and StandardScaler Pickle
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open("models/scaler.pkl","rb"))


# Route for HomePage
@app.route('/')
def index():
    return render_template('index.html')

# Route for HomePage
# Handles both 'GET' and 'POST'
@app.route('/predictdata', methods=['GET','POST']) 
def predict_datapoint():
    # Will read values from 'home.html' and 
    # Will send values from model to 'Home.html'
    
    if request.method == 'POST':
        Temprature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region')) 
        
        # Use data in the order for model training
        new_data_scaled = standard_scaler.transform([[Temprature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)    

        return render_template('home.html', result=result[0])
   
    else:
        # For Get Request
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")
# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'Reg_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = '🎉🎉Congrats You are Fit🎉🎉' if prediction[0] >= 0.5 else '🥺🥺You are not Fit🥺🥺'

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
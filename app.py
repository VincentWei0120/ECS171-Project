from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and transformers
model = pickle.load(open('housing.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
poly_features = pickle.load(open('poly_features.pkl', 'rb'))

# Define input features
input_features = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        input_data = [float(request.form.get(feature)) for feature in input_features]

        # Convert to DataFrame for consistency
        input_df = pd.DataFrame([input_data], columns=input_features)

        # Scale and transform the input data
        input_scaled = scaler.transform(input_df)
        input_poly = poly_features.transform(input_scaled)

        # Make prediction using the model
        prediction = model.predict(input_poly)

        # Format the prediction for display
        predicted_value = prediction[0]

        return render_template("result.html", prediction=predicted_value)

    else:
        return "Something went wrong. Please try again."

if __name__ == "__main__":
    app.run(debug=True)

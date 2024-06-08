# ECS171-Project

README for Running the Flask Application
This guide will help you set up a virtual environment and run the Flask application for predicting house prices in California.

Prerequisites
Python 3 installed
Basic understanding of Python, HTML, and Flask
Instructions
Clone or download the project files

Ensure you have the following files in your project directory:
app.py
index.html
result.html
housing.pkl
scaler.pkl
poly_features.pkl
housing.csv (for reference)

Create and activate a virtual environment

Open your terminal or command prompt, navigate to your project directory, and create a virtual environment:

python -m venv venv

Activate the virtual environment:

For Windows:
venv\Scripts\activate.bat

For macOS/Linux:
source venv/bin/activate

Install required packages
Inside the activated virtual environment, install Flask and other necessary packages:
pip install Flask pandas numpy scikit-learn

Run the Flask application

With the virtual environment still activated, run the Flask application:
python app.py

Open the application in your browser
Open your web browser and navigate to http://127.0.0.1:5000/. You should see the input form for the house price prediction.

Explanation
app.py: The main Flask application file.
index.html: The HTML template for the input form.
result.html: The HTML template for displaying prediction results.
housing.pkl, scaler.pkl, poly_features.pkl: Pickle files containing the trained model and preprocessing transformers.
Follow these steps, and you should be able to run the Flask application and predict house prices based on user input.

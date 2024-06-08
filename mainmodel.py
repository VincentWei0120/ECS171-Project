import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


data = pd.read_csv('housing.csv')
df = data.dropna()

columns = ['housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value']

Q1 = df[columns].quantile(0.25)
Q3 = df[columns].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[columns] < (Q1 - 1.5*IQR))| (df[columns] > (Q3+1.5*IQR))).any(axis=1)
df_cleaned = df[-outliers]

categorical_features = [col for col in df_cleaned.columns if df_cleaned[col].dtype == 'object']
X = pd.get_dummies(df_cleaned, columns=categorical_features, drop_first=True)
y = df_cleaned['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X.drop('median_house_value', axis=1), y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

# Using Ridge Regression for regularization
ridge_reg = Ridge(alpha=1, solver="cholesky")
model = ridge_reg.fit(X_train_poly, y_train)
y_pred_ridge = ridge_reg.predict(X_test_poly)

pickle.dump(model, open('housing.pkl', 'wb'))
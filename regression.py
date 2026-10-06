import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Size": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
    "Age": [10, 8, 7, 6, 5, 4, 3, 2, 2, 1],
    "Price": [50, 58, 72, 82, 95, 105, 120, 135, 145, 155]
}

df = pd.DataFrame(data)

print(df)

X = df[["Size", "Bedrooms", "Age"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)

print("\nMultiple Linear Regression")
print("Predictions:", linear_prediction)
print("R2 Score:", r2_score(y_test, linear_prediction))


polynomial_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

polynomial_model.fit(X_train, y_train)

polynomial_prediction = polynomial_model.predict(X_test)

print("\nPolynomial Regression")
print("Predictions:", polynomial_prediction)
print("R2 Score:", r2_score(y_test, polynomial_prediction))


ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)

ridge_prediction = ridge_model.predict(X_test)

print("\nRidge Regression")
print("Predictions:", ridge_prediction)
print("R2 Score:", r2_score(y_test, ridge_prediction))


lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

lasso_prediction = lasso_model.predict(X_test)

print("\nLasso Regression")
print("Predictions:", lasso_prediction)
print("R2 Score:", r2_score(y_test, lasso_prediction))


print("\nMean Squared Error")

print("Multiple Linear Regression:",
      mean_squared_error(y_test, linear_prediction))

print("Polynomial Regression:",
      mean_squared_error(y_test, polynomial_prediction))

print("Ridge Regression:",
      mean_squared_error(y_test, ridge_prediction))

print("Lasso Regression:",
      mean_squared_error(y_test, lasso_prediction))
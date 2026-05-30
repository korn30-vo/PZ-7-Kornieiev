import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("automobile.csv", encoding="latin1")

data = df[['engine-size','highway-mpg','normalized-losses','price']]
data = data.apply(pd.to_numeric, errors='coerce').dropna()

y = data['price']

# SLR
X1 = data[['engine-size']]
slr = LinearRegression().fit(X1, y)
slr_pred = slr.predict(X1)

# MLR
X2 = data[['normalized-losses','highway-mpg']]
mlr = LinearRegression().fit(X2, y)
mlr_pred = mlr.predict(X2)

# POLY
X3 = data[['highway-mpg']]
poly = PolynomialFeatures(degree=5)
X3_poly = poly.fit_transform(X3)

poly_model = LinearRegression().fit(X3_poly, y)
poly_pred = poly_model.predict(X3_poly)

print("SLR R2:", r2_score(y, slr_pred))
print("MLR R2:", r2_score(y, mlr_pred))
print("POLY R2:", r2_score(y, poly_pred))
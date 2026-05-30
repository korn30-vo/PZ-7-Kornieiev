import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("automobile.csv", encoding="latin1")

data = df[['highway-mpg', 'price']].copy()
data = data.apply(pd.to_numeric, errors='coerce').dropna()

X = data[['highway-mpg']]
y = data['price']

poly = PolynomialFeatures(degree=11)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

print("R2:", r2_score(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))

# ===== PLOT =====
X_grid = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
X_grid_poly = poly.transform(X_grid)

plt.scatter(X, y)
plt.plot(X_grid, model.predict(X_grid_poly), color='red')
plt.title("Polynomial Regression (degree 11)")
plt.show()
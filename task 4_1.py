import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

df = pd.read_csv("automobile.csv", encoding="latin1")

data = df[['highway-mpg','price']].copy()
data = data.apply(pd.to_numeric, errors='coerce').dropna()

X = data[['highway-mpg']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

poly = PolynomialFeatures(degree=15)

X_train_p = poly.fit_transform(X_train)
X_test_p = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_p, y_train)

print("Train R2:", model.score(X_train_p, y_train))
print("Test R2:", model.score(X_test_p, y_test))

# ===== VISUALIZATION =====
plt.scatter(X_train, y_train, label="Train")
plt.scatter(X_test, y_test, label="Test")
plt.title("Overfitting")
plt.legend()
plt.show()
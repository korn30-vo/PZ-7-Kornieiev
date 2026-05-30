import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("automobile.csv", encoding="latin1")

data = df[['normalized-losses', 'highway-mpg', 'price']].copy()
data = data.apply(pd.to_numeric, errors='coerce').dropna()

X = data[['normalized-losses', 'highway-mpg']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("R2:", r2_score(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))

# ===== DISTRIBUTION =====
plt.figure(figsize=(8,5))
sns.kdeplot(y, label="Actual", fill=True)
sns.kdeplot(y_pred, label="Predicted", fill=True)
plt.legend()
plt.title("Distribution Comparison")
plt.show()
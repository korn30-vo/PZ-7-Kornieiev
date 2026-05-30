import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


poly = PolynomialFeatures(degree=1)

X_train_p = poly.fit_transform(X_train)
X_test_p = poly.transform(X_test)


model = LinearRegression()
model.fit(X_train_p, y_train)


print("Train R2:", model.score(X_train_p, y_train))
print("Test R2:", model.score(X_test_p, y_test))

plt.scatter(X_train, y_train, color="blue", label="Train", alpha=0.7)
plt.scatter(X_test, y_test, color="green", label="Test", alpha=0.7)


X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
X_plot_p = poly.transform(X_plot)
y_plot = model.predict(X_plot_p)

plt.plot(X_plot, y_plot, color="red", linewidth=2, label="Prediction")


plt.title("Underfitting (Degree = 1)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
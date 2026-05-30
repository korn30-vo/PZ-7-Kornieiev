import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("automobile.csv")

df = df[['normalized-losses', 'highway-mpg', 'price']]

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.fillna(df.median(numeric_only=True))

X = df[['normalized-losses', 'highway-mpg']]
y = df['price']


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])


params = {
    'ridge__alpha': [0.01, 0.1, 1, 10, 100]
}

cv_value = min(5, len(df))

grid = GridSearchCV(
    pipeline,
    params,
    cv=cv_value,
    scoring='neg_mean_squared_error'   # 👈 ВАЖЛИВО
)

grid.fit(X, y)

print("Best alpha:", grid.best_params_)
print("Best score:", grid.best_score_)
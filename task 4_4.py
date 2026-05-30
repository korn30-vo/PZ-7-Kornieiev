import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer


df = pd.read_csv("automobile.csv", encoding="latin1")

data = df[['normalized-losses', 'highway-mpg', 'price']].copy()


for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna()

X = data[['normalized-losses', 'highway-mpg']]
y = data['price']


imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)


model = Ridge(alpha=1)


cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)

scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

print("Samples:", len(X))
print("Scores:", scores)
print("Mean R2:", scores.mean())
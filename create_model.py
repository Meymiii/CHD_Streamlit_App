import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

print("Creation du modele en cours...")

np.random.seed(42)
data = {
    'sbp': np.random.uniform(100, 200, 100),
    'ldl': np.random.uniform(2, 8, 100),
    'adiposity': np.random.uniform(15, 40, 100),
    'obesity': np.random.uniform(20, 40, 100),
    'age': np.random.randint(30, 70, 100),
    'famhist': np.random.choice(['Present', 'Absent'], 100),
    'chd': np.random.randint(0, 2, 100)
}

df = pd.DataFrame(data)
X = df.drop('chd', axis=1)
y = df['chd']

numeric_features = ['sbp', 'ldl', 'adiposity', 'obesity', 'age']
categorical_features = ['famhist']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=3)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

print("Entrainement du modele...")
pipeline.fit(X, y)

joblib.dump(pipeline, 'Model.pkl')

print("Modele cree avec succes dans Model.pkl")
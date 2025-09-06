# src/main.py

import pandas as pd
from src.preprocessing import create_preprocessor
from src.train_models import train_and_evaluate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("../data/allsides_balanced_news_headlines-texts.csv")

# Define features and target
X = df.drop("bias_rating", axis=1)
y = df["bias_rating"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode target
le = LabelEncoder()
y_encoded_train = le.fit_transform(y_train)
y_encoded_test = le.transform(y_test)

# Define column types
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = ["source"]
text_col = "text"



# Preprocessing
preprocessor = create_preprocessor(num_cols, cat_cols, text_col)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Train and evaluate models
train_and_evaluate(X_train_preprocessed, X_test_preprocessed, y_train, y_test)
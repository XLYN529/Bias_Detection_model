# 📰 Political Bias Text Classification

This project builds a **multi-class text classification model** that predicts whether a news article leans **center**, **left**, or **right**.  
The workflow includes data preprocessing, TF-IDF feature extraction, model training, and evaluation using multiple classifiers.

---

## 📂 Project Structure

```
project/
├── data/                # Raw and processed datasets (not included in repo if large)
├── notebooks/           # Jupyter notebooks used for EDA, experiments, and model development
├── scipts/                 # Source code for preprocessing, training, and evaluation
│   ├── preprocessing.py # Custom preprocessing and feature engineering code
│   ├── train_models.py         # Script to train models (Logistic Regression, Random Forest, XGBoost)
├── models/              # Saved trained models (pickle/joblib files)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore file
```

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/political-bias-classification.git
cd Bias_Detection_model
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## 🧹 Data Cleaning & Preprocessing

- Handled missing values in `source` and `text` columns  
- Replaced empty tag lists with `"NoTag"`  
- Removed extreme outliers after analysis  
- Standardized text to lowercase  
- Applied **TF-IDF vectorization** on the `text` column  
- Encoded categorical labels (`center`, `left`, `right`) using `LabelEncoder`  

---

## ✨ Feature Extraction

**TF-IDF Vectorization** was used with the following parameters:  
- `max_features=5000`  
- `stop_words="english"`  
- `ngram_range=(1,2)` (unigrams + bigrams)  

---

## 🤖 Models Trained

- **Logistic Regression** (best performer)  
- **Random Forest**  
- **XGBoost**  

---

## 📈 Results

| Model               | Accuracy | Macro F1 |
|----------------------|----------|----------|
| Logistic Regression | **0.99** | **0.99** |
| Random Forest       | 0.96     | 0.95     |
| XGBoost             | 0.98     | 0.98     |

**Confusion Matrix (Logistic Regression)**  
```
[[ 841    8    1]
 [   9 2045    1]
 [   9    5 1430]]
```

---

## 🚀 Usage

Train models:
```bash
python scripts/train.py
```

Make predictions:
```python
from scripts.pipeline import predict_text

print(predict_text("The government passed a new healthcare bill."))
# -> "center"
```

---

## 💾 Save & Load Model

```python
import pickle

# Save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
```

---

## 📌 Future Improvements

- Experiment with deep learning models (BERT, LSTMs)  
- Deploy using Flask or FastAPI  
- Build a frontend demo for live predictions  

---

## 👤 Author

Developed by Shariq Abdul Aziz (https://github.com/XLYN529)  
⭐ Star this repo if you found it useful!


# 🔍 Malicious URL Detection - Korea University CURT Project

This project was developed as part of the CURT (Creative Undergraduate Research Training) program at Korea University.  
It focuses on detecting **malicious URLs** using feature-based machine learning and ensemble techniques.

---

## 📌 Project Overview

- 🔒 **Goal**: Classify URLs as malicious or benign based on structural features
- 🧠 **Model**: XGBoost-based ensemble with K-Fold cross-validation
- ⚙️ **Frameworks**: Python, Scikit-learn, XGBoost, Flask (for demo)
- 💾 **Data**: Custom dataset with engineered features from URLs (length, subdomain count, special character count, etc.)

---

## 🗂️ Directory Structure

```
ku-checkurl/
│
├── app.py                  # Flask web app for real-time URL classification
├── train_model.py          # Training script for the ML model
├── train.csv               # Labeled training data
├── malicious_url_models.pkl  # Trained ensemble model
├── templates/
│   └── index.html          # Front-end for the web app
└── app_log_x.py            # Optional logging module
```

---

## 🛠️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> If not available, manually install: `xgboost`, `scikit-learn`, `flask`, `pandas`, `tldextract`

### 2. Run the app

```bash
python app.py
```

The app will launch on `http://localhost:5000`

---

## 🌐 Feature Engineering

The following features were extracted from raw URLs:

- URL length
- Subdomain count
- Count of special characters (`-`, `_`, `/`)
- (Optionally expandable with lexical or host-based features)

---

## 📊 Model Performance

- Model: `XGBClassifier` with 4-fold ensemble
- Metric: ROC-AUC
- Average AUC across folds: **(insert value here after training)**

---

## 🙌 Contributors

- **양진영** - Korea Military Academy / Korea University CURT Program
- Project guidance: Korea University AI Lab

---

## 📄 License

This project is for academic research and education purposes.

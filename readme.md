# Breast Cancer Cytology Classifier



A clean, modular pipeline for classifying fine‑needle aspirate (FNA) breast‑cancer cytology samples (malignant vs. benign) using logistic regression on the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

---

## 🎯 Key Features

- **Modular Code Organization**: Reusable utilities under `src/` for data loading, modeling, and visualization.
- **Reproducible Environment**: `requirements.txt` and editable install (`pip install -e .`) ensure consistent setup.
- **Interactive Analysis**: Jupyter notebook demonstrating end‑to‑end workflow (EDA, training, evaluation, feature importance).
- **Results Tracking**: All plots and reports saved under `results/` for easy review and sharing.

---

## 📁 Repository Structure

```text
breast-cancer-cytology-classifier/
├── data/
│   └── uci_breast_cancer_diagnostic.csv   # Sample CSV of the Diagnostic FNA dataset
├── notebooks/
│   └── breast_cancer_cytology_classifier.ipynb  # Main analysis notebook
├── results/                               # Generated figures & metrics (PNG, CSV)
├── src/                                   # Modular Python package
│   ├── __init__.py                        # Package marker
│   ├── data_utils.py                     # Data loading & splitting
│   ├── modeling.py                       # Model training & evaluation
│   └── viz.py                            # Plotting utilities
├── requirements.txt                      # Exact pip dependencies
├── setup.py                              # Editable install configuration
└── README.md                             # Project overview & instructions
```

---

## 🚀 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/Ozai-03/breast-cancer-cytology-classifier.git
   cd breast-cancer-cytology-classifier
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install as editable package**

   ```bash
   pip install -e .
   ```

---

## 🎓 Usage

### 1. Notebook Analysis

Launch the Jupyter notebook to run the full pipeline:

```bash
jupyter notebook notebooks/breast_cancer_cytology_classifier.ipynb
```

Follow the documented sections:

- **Load & Split Data**
- **Exploratory Data Analysis** (Heatmap & Pairplot)
- **Model Training & Evaluation**
- **Feature Importance**
- **Conclusions & Next Steps**

### 2. Package API

You can also import and use the modules directly in Python scripts:

```python
from src.data_utils import load_data, split_data
from src.modeling   import train_model, evaluate_model, get_coefficients, sort_coefficients_abs
from src.viz        import plot_heatmap, plot_feature_importance

# Example
import pandas as pd

df = load_data('data/uci_breast_cancer_diagnostic.csv')
X_train, X_test, y_train, y_test = split_data(df)
model = train_model(X_train, y_train)
cm, report = evaluate_model(model, X_test, y_test)
print(report)
```

---

## 📊 Results

- **Heatmap**: `results/heatmap.png`
- **Pairplot**: `results/pairplot.png`
- **Classification Report**: `results/classification_report.csv` 
- **Feature Importance**: `results/feature_importance.png`

---

## 🤝 Contributing

Contributions welcome! Please fork the repo and submit a pull request for enhancements, bug fixes, or additional features.

---

*Last updated: 2025-07-01*


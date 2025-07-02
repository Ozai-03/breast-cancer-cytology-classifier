# Data Directory

This folder contains both the full UCI Breast Cancer Wisconsin (Diagnostic) FNA cytology dataset and a smaller sample subset for quick prototyping.

## Files

- **uci\_breast\_cancer\_diagnostic.csv**\
  The complete Diagnostic FNA dataset (569 samples × 31 columns: 30 features + `target`).

- **sample\_uci\_breast\_cancer\_diagnostic.csv**\
  A sample subset of 100 rows from the full dataset, for rapid testing and demonstration.

## Loading Data

The `load_data()` helper in `src/data_utils.py` automatically locates and loads the full dataset:

```python
from src.data_utils import load_data

df = load_data()
print(df.shape)   # Expect (569, 31)
print(df.head())
```

## Splitting Data

Use the `split_data()` function to split features and target into training and test sets:

```python
from src.data_utils import split_data

X_train, X_test, y_train, y_test = split_data(
    df,
    test_size=0.2,
    random_state=33
)
print(X_train.shape, X_test.shape)
```

## Why Include a Sample Subset?

- **Lightweight:** Keeps cloning and initial setup fast.
- **Quick Start:** Run notebooks end-to-end without loading all 569 samples.
- **Consistency:** Ensures everyone can reproduce examples using the same fixed subset.

---

*Place this **`README.md`** in your **`data/`** folder to document its contents and usage.*


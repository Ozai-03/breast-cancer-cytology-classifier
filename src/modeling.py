import pandas as pd
import shap
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from typing import Any


def train_model(x_train: pd.DataFrame,
                y_train: pd.Series,
                solver: str = 'liblinear') -> LogisticRegression:
    
    """
    Train a logistic regression model.
    
    Args:
        x_train: Training features
        y_train: Training labels
        solver: Solver algorithm for optimization
        
    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(solver=solver)
    model.fit(x_train, y_train)
    return model

def evaluate_model(model: LogisticRegression,
                   x_test: pd.DataFrame,
                   y_test: pd.Series) -> tuple[Any, dict]:
    """
    Evaluate model performance.
    
    Args: 
        model: Trained classifier
        x_test: Test features
        y_test: True test lables
        
    Returns:
        confusion matric and classifcation report as dict
    """

    preds = model.predict(x_test)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    return cm, report


def get_coefficients(model: LogisticRegression,
                     feature_names: list) -> pd.DataFrame:
    """
    Extract model coefficients as a DataFrame
    
    Args:
    model: Trained LogisticRegression
    feature_names: List of feature column names
    
    Returns:
        DataFrame (1 x n_features) of coefficients"""
    return pd.DataFrame(model.coef_, columns=feature_names)

def normalize_coefficiets(coefficients: pd.DataFrame,
                          x: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize coefficients by feature standard deviation.
    
    Args:
        coefficients: DataFrame of model coefficients
        x: Original feature DataFrame used for training/testing
        
    Returns:
        DataFrame of normalized coefficients
    """
    return coefficients.div(x.std())
    
def sort_coeffcients_abs(coefficients: pd.DataFrame) -> pd.Series:
    """
    Sort coefficients by absolute value descending
    
    Args:
        coefficients: DataFrame of model coefficients
        
    Returns:
        Series of coefficients sorted by absolute value"""
    abs_df = coefficients.abs()
    sorted_df = abs_df.sort_values(by=0, axis=1, ascending=False)
    return sorted_df.iloc[0]

def shap_explainer(model: LogisticRegression,
                   x_train: pd.DataFrame,
                   x_test: pd.DataFrame,
                   bg_size: int = 1000,
                   ts_size: int = 500,
                   random_state: int = 42
                   ) -> tuple:
    """
    Gets feature importance using mean absolute SHAP values.
    
    Args:
        model: Trained model with predict method
        x_train: Training features (DataFrame)
        x_test: Test Features (DataFrame)
        bg_size: Max number of test samples to explain
        random_state: Seed for reproducibility

    Returns:
        DataFrame sorted by descending importance with columns ['feature, 'importance']
        """
    #subsample
    bg = shap.sample(x_train, bg_size,random_state=random_state)
    ts = shap.sample(x_test, ts_size,random_state=random_state)
    
    #compute SHAP values
    explainer = shap.Explainer(model.predict,bg)
    shap_vals = explainer(ts).values 

    importances = np.mean(np.abs(shap_vals), axis=0)

    #sort features by importance
    sorted_indicies = np.argsort(importances)[::1]
    feature_names = x_test.columns[sorted_indicies]
    importance_values = importances[sorted_indicies]
    importance_df = pd.DataFrame({'feature':feature_names, 'importance': importance_values})
    return (importance_df, shap_vals, ts)



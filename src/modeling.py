import pandas as pd
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
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data() -> pd.DataFrame:
    r"""
    Load the UCI Breast Cancer Wisconsin (Diagnostic) dataset into a DataFrame
    
    Args:
        csv_path: Path to the CSV file containing the dataset (default: 'data\uci_breast_cancer_diagnostic.csv')

    Returns:
        DataFrame with feature columns and 'target' column.
    """

    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_dir, 'data', 'uci_breast_cancer_diagnostic.csv')

    df = pd.read_csv(path)
    return df
    
def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 33):
    """
    Split DataFrame into training and test sets.
    
    Args:
        df: DataFrame including features and 'target'.
        test_size: Fraction of data to reserve for testing.
        random_state: Seed for reproducibility.
        
    Returns:
        x_train, x_test, y_train, y_test
    """

    x = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    
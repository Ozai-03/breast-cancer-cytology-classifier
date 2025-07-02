import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_pairplot(df: pd.DataFrame, features: list, hue: str = 'target') -> sns.PairGrid:
    """
    Plot a pairplot for the specified features colored by hue and return the PairGrid.

    Args:
        df: DataFrame containing data
        features: List of column names to include
        hue: Column name for coloring (default: 'target')

    Returns:
        Seaborn PairGrid object for further customization or saving.
    """
    grid = sns.pairplot(df, vars=features, hue=hue)
    return grid


def plot_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Plot a correlation heatmap for the DataFrame and return the Figure.

    Args:
        df: DataFrame containing data

    Returns:
        Matplotlib Figure object with the heatmap.
    """
    fig, ax = plt.subplots(figsize=(25, 30))
    sns.heatmap(df.corr(), annot=True, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    return fig


def plot_feature_importance(coef_series: pd.Series) -> plt.Figure:
    """
    Plot a barplot of feature importances from a coefficient series and return the Figure.

    Args:
        coef_series: Series indexed by feature names

    Returns:
        Matplotlib Figure object with the feature importance barplot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=coef_series.values, y=coef_series.index, ax=ax)
    ax.set_xlabel('Coefficient Value')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance')
    return fig
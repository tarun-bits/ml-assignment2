import pandas as pd


def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate entries from the dataset."""
    return data.drop_duplicates()
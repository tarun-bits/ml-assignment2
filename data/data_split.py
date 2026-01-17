# This is standalone code for splitting a dataset into training and test sets.
# Test set will be saved in a separate CSV file and will be used to upload in streamlit application to run validation
# training set will be used to train the model and we will split this training set further into training and validation sets for model evaluation
# Usage: right click on this file in IDE and hit run
from sklearn.model_selection import train_test_split
import pandas as pd



def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """Split the dataset into training, validation, and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split to get test set
    X_sub, X_test, y_sub, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Save test set to CSV

    test_df = X_test.copy()
    test_df[target_col] = y_test
    test_df.to_csv("./ObesityDataSet_test.csv", index=False)

    sub_df = X_sub.copy()
    sub_df[target_col] = y_sub
    sub_df.to_csv("./ObesityDataSet_train.csv", index=False)



df = pd.read_csv("./original_data/ObesityDataSet_raw_and_data_sinthetic.csv")
target_col = "NObeyesdad"
split_data(df, target_col)
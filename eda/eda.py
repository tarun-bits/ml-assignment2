# eda module is responsible for loading eda set and performing exploratory eda analysis (EDA) tasks
from copy import deepcopy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

class Dataset:

    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}")
            self.data = data
        except Exception as e:
           raise e

    @property
    def data_frame(self) -> pd.DataFrame:
        """Get the loaded DataFrame."""
        return self.data

    def non_numerical_columns(self, exclude_target: bool = False) -> list:
        """Get a list of non-numerical (categorical) columns in the DataFrame."""
        if self.data is None:
            print("No data loaded.")
            return []
        data = deepcopy(self.data)
        if exclude_target:
            data = data.drop(columns=[self.get_target_column])

        non_numerical_cols = data.select_dtypes(exclude=['float64']).columns.tolist()
        return non_numerical_cols


    def numerical_columns(self, exclude_target: bool = False) -> list:
        """Get a list of non-numerical (categorical) columns in the DataFrame."""
        if self.data is None:
            print("No data loaded.")
            return []
        data = deepcopy(self.data)
        if exclude_target:
            data = data.drop(columns=[self.get_target_column])
        non_numerical_cols = data.select_dtypes(exclude=['object']).columns.tolist()
        return non_numerical_cols

    @property
    def get_target_column(self) -> str:
        """Get a list of target columns in the DataFrame."""
        if self.data is None:
            print("No data loaded.")
            return []
        return "NObeyesdad"


    def visualize_data(self):
        """Visualize the DataFrame using basic plots."""
        if self.data is None:
            print("No data loaded.")
            return

        plt.figure(figsize=(8, 4))
        sns.countplot(data=self.data_frame, x=self.get_target_column)
        plt.title("Target Class Distribution")
        plt.xticks(rotation=45)
        plt.show()

        cols = 2
        num_cols = len(self.numerical_columns())
        if num_cols == 0:
            return
        rows = int(np.ceil(num_cols / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), squeeze=False)
        axes_flat = axes.flatten()

        for i, col in enumerate(self.numerical_columns()):
            ax = axes_flat[i]
            sns.boxplot(data=self.data_frame, x=self.get_target_column, y=col, ax=ax)
            ax.set_title(f"{col} vs {self.get_target_column}")
            ax.tick_params(axis='x', rotation=45)

        # hide any unused subplots
        for j in range(num_cols, rows * cols):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(15, 6))
        sns.boxplot(data=self.data_frame[self.numerical_columns()])
        plt.xticks(rotation=90)
        plt.title("Outlier Detection in Numerical Features")
        plt.show()

    def update_data_frame(self, new_data: pd.DataFrame):
        """Update the DataFrame with new data.

        Args:
            new_data: A pandas DataFrame to replace the current data.
        """
        self.data = new_data
        print("DataFrame updated successfully.")

    def show_data(self):
        """Print the first 5 rows of the DataFrame."""
        if self.data is None:
            print("No data loaded.")
            return
        print(f"First 5 rows of {self.file_path}")
        print(self.data.head(5))

    def show_unique_values(self):
        """Print unique values for non-numeric (categorical) columns only."""
        if self.data is None:
            print("No data loaded.")
            return

        print("Unique values for non-numeric (categorical) columns:\n")

        non_numeric_cols = self.data.select_dtypes(exclude=['float64']).columns
        if len(non_numeric_cols) == 0:
            print("No non-numeric (categorical) columns found.")
            return

        for col in non_numeric_cols:
            print(f"Unique values in column '{col}': {self.data[col].unique()}")


    def summarize_data(self):
        """Summarize the DataFrame by providing basic statistics and information.

        Parameters:
        eda (pd.DataFrame): The DataFrame to summarize.

        Returns:
        None
        """
        if self.data is not None:
            print("Data Summary:")
            print(self.data.info())
            print("\nStatistical Summary:")
            print(self.data.describe())
            print("\nMissing Values:")
            print(self.data.isnull().sum())
            print(f"Duplicate data present: {self.data.duplicated().sum() != 0}")
            if self.data.duplicated().sum() != 0:
                print(f"Number of duplicate rows: {self.data.duplicated().sum()}")
                print("Duplicate rows:")
                print(self.data[self.data.duplicated()])

        else:
            print("No eda to summarize.")

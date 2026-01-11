# eda module is responsible for loading eda set and performing exploratory eda analysis (EDA) tasks

import pandas as pd

class Dataset:

    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}")
            self.data = data
        except Exception as e:
           raise e

    def show_data(self):
        print(self.data.head(5))

    def show_unique_values(self):
        """loops through all columns in self.data and print unique values for each column
        """

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
        else:
            print("No eda to summarize.")

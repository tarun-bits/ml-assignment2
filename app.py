from eda import eda
from preprocessor.drop_duplicates import drop_duplicates

if __name__ == "__main__":
    # Load the data and perform eda
    dataset = eda.Dataset("data/ObesityDataSet_raw_and_data_sinthetic.csv")
    dataset.summarize_data()
    dataset.show_data()
    dataset.show_unique_values()

    # Basis EDA duplicate values found
    cleaned_df = drop_duplicates(dataset.data_frame)

    # updating data frame to recheck if duplicates are removed
    dataset.update_data_frame(cleaned_df)

    dataset.summarize_data()
    dataset.show_data()
    dataset.show_unique_values()
    dataset.visualize_data()


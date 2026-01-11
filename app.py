from eda import eda

if __name__ == "__main__":
    # Load the data
    dataset = eda.Dataset("data/ObesityDataSet_raw_and_data_sinthetic.csv")
    dataset.summarize_data()
    dataset.show_data()
    dataset.show_unique_values()
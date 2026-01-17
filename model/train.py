from eda import eda
from helpers import TrainTestSplit
from model.logistic_regression import model as lr_model
from preprocessor.drop_duplicates import drop_duplicates
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __name__ == "__main__":
    # Load the data and perform eda
    dataset = eda.Dataset("../data/ObesityDataSet_train.csv")
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
    # dataset.visualize_data()

    X = dataset.data_frame.drop(columns=[dataset.get_target_column])
    y = dataset.data_frame[dataset.get_target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    trainTestSplit = TrainTestSplit(X_train, X_test, y_train, y_test)

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, dataset.numerical_columns(exclude_target=True)),
            ("cat", categorical_transformer, dataset.non_numerical_columns(exclude_target=True)),
        ],
        remainder="drop"
    )

    # Create Logistic Regression model
    logistic_regression_pipeline = lr_model.LogisticRegressionPipeline(
        preprocessor=preprocessor,
        train_test_split=trainTestSplit
    )
    model = logistic_regression_pipeline.train_model()
    accuracy = logistic_regression_pipeline.evaluate_model(model)
    print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")
    logistic_regression_metrics = logistic_regression_pipeline.model_metrics(model)
    print(f"Logistric Regression Model Metrics: {logistic_regression_metrics}" )
    logistic_regression_pipeline.save_pkl_model(model)
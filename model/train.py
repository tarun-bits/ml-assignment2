from eda import eda
from helpers import TrainTestSplit
from model.logistic_regression import model as lr_model
from model.decision_tree_classifier import model as dtc_model
from model.ensemble_xgboost import model as xgboost_model
from model.ensemble_random_forest import model as random_forest_model
from model.knn_classifier import model as knn_model
from model.naive_bayes_gaussian import model as gnb_model
from preprocessor.drop_duplicates import drop_duplicates
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

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

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

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

    # Create Decision Tree model
    decision_tree_pipeline = dtc_model.DecisionTreeClassifierPipeline(
        preprocessor=preprocessor,
        train_test_split=trainTestSplit
    )
    model = decision_tree_pipeline.train_model()
    accuracy = decision_tree_pipeline.evaluate_model(model)
    print(f"Decision Tree Model Accuracy: {accuracy:.4f}")
    decision_tree_metrics = decision_tree_pipeline.model_metrics(model)
    print(f"Decision Tree Model Metrics: {decision_tree_metrics}" )
    decision_tree_pipeline.save_pkl_model(model)

    # KNN Classifier
    knn_pipeline = knn_model.KnnClassifierPipeline(
        preprocessor=preprocessor,
        train_test_split=trainTestSplit
    )
    model = knn_pipeline.train_model()
    accuracy = knn_pipeline.evaluate_model(model)
    print(f"KNN Model Accuracy: {accuracy:.4f}")
    knn_metrics = knn_pipeline.model_metrics(model)
    print(f"KNN Metrics: {knn_metrics}" )
    knn_pipeline.save_pkl_model(model)


    # Naive Bayes Gaussian Classifier
    gaussian_nb_pipeline = gnb_model.NaiveBayesGaussian(
        preprocessor=preprocessor,
        train_test_split=trainTestSplit
    )
    model = gaussian_nb_pipeline.train_model()
    accuracy = gaussian_nb_pipeline.evaluate_model(model)
    print(f"GNB Model Accuracy: {accuracy:.4f}")
    gnb_metrics = gaussian_nb_pipeline.model_metrics(model)
    print(f"GNB Metrics: {gnb_metrics}" )
    gaussian_nb_pipeline.save_pkl_model(model)

    # Random Forest Ensemble Classifier
    random_forest_pipeline = random_forest_model.RandomForestModel(
        preprocessor=preprocessor,
        train_test_split=trainTestSplit
    )
    model = random_forest_pipeline.train_model()
    accuracy = random_forest_pipeline.evaluate_model(model)
    print(f"Ensemble Random Forest Model Accuracy: {accuracy:.4f}")
    rndf_metrics = random_forest_pipeline.model_metrics(model)
    print(f"Ensemble Random Forest Metrics: {rndf_metrics}" )
    random_forest_pipeline.save_pkl_model(model)

    # XGBoost Ensemble Classifier
    xgboost_model_pipeline = xgboost_model.XGBoostEnsembleModel(
        preprocessor=preprocessor,
        train_test_split=trainTestSplit
    )
    model = xgboost_model_pipeline.train_model()
    accuracy = xgboost_model_pipeline.evaluate_model(model)
    print(f"Ensemble XGBoost Model Accuracy: {accuracy:.4f}")
    xgb_metrics = xgboost_model_pipeline.model_metrics(model)
    print(f"Ensemble XGBoost Model Metrics: {xgb_metrics}" )
    xgboost_model_pipeline.save_pkl_model(model)
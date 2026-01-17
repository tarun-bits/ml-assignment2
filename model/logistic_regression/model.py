from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from helpers import TrainTestSplit


class LogisticRegressionPipeline():
    def __init__(self, preprocessor: ColumnTransformer, train_test_split: TrainTestSplit):
        self.model_lib = LogisticRegression(max_iter=10000, C=10)
        self.preprocessor = preprocessor
        self.trainTestSplit = train_test_split

    def setup_pipeline(self):
        model = Pipeline(steps=[
            ("preprocess", self.preprocessor),
            ("clf", self.model_lib)
        ])
        return model

    def train_model(self):
        model = self.setup_pipeline()
        model.fit(
            self.trainTestSplit.X_train,
            self.trainTestSplit.y_train
        )
        return model

    def evaluate_model(self, model):
        accuracy = model.score(
            self.trainTestSplit.X_test,
            self.trainTestSplit.y_test
        )
        return accuracy

    def model_metrics(self, model):
        y_pred = model.predict(self.trainTestSplit.X_test)
        y_pred_proba = model.predict_proba(self.trainTestSplit.X_test)
        cm = confusion_matrix(self.trainTestSplit.y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
        cm_display.plot()
        plt.show()
        accuracy = accuracy_score(self.trainTestSplit.y_test, y_pred)
        precision = precision_score(self.trainTestSplit.y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(self.trainTestSplit.y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(self.trainTestSplit.y_test, y_pred, average="weighted", zero_division=0)
        mcc_score = matthews_corrcoef(self.trainTestSplit.y_test, y_pred)
        print(f"MCC Score: {mcc_score}")

        auc_ovr = roc_auc_score(self.trainTestSplit.y_test, y_pred_proba, multi_class='ovr')
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "AUC": auc_ovr, "MCC": mcc_score}

    def save_pkl_model(self, model):
        file_path = Path(__file__).with_name('logistic_regression_model.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
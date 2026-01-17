import pickle

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, roc_auc_score

from eda.eda import Dataset


class model_evaluate:
    def __init__(self, pkl_file_path: str, dataset: Dataset):
        self.pkl_file_path = pkl_file_path
        self.dataset = dataset

    def is_valid(self) -> bool:
        try:
            with open(self.pkl_file_path, "rb") as f:
                pickle.load(f)
            return True
        except Exception:
            return False

    def evaluate(self):
        model = None
        with open(self.pkl_file_path, "rb") as f:
            model = pickle.load(f)
        if model is not None:
            try:
                ds = self.dataset.data
                X_test = ds.drop(columns=[self.dataset.get_target_column])
                y_test = ds[self.dataset.get_target_column]
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                cm = confusion_matrix(y_test, y_pred)
                cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
                cm_display.plot()
                plt.show()
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                mcc_score = matthews_corrcoef(y_test, y_pred)
                print(f"MCC Score: {mcc_score}")

                auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "AUC": auc_ovr,
                        "MCC": mcc_score}, cm
            except Exception as e:
                raise e
        else:
            raise ValueError("Model could not be loaded from the provided pickle file.")


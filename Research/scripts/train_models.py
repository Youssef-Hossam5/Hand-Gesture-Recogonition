import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import pandas as pd

def train_all_models(X, y, experiment_name="Hand Gestures", save_path="models/"):
    # Create save path if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # set tracking url
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # ----------------- RANDOM FOREST -----------------
    with mlflow.start_run(run_name="RandomForest"):
        rf_params = {"n_estimators": 200, "random_state": 42}
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        acc_rf = accuracy_score(y_test, y_pred_rf)
        print("RF Accuracy:", acc_rf)

        # Log parameters and metrics
        mlflow.log_params(rf_params)
        mlflow.log_metric("accuracy", acc_rf)

        # Log classification report
        report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
        report_df = pd.DataFrame(report_rf).transpose()
        report_path = os.path.join(save_path, "rf_classification_report.csv")
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)

    # ----------------- SVM -----------------
    with mlflow.start_run(run_name="SVM"):
        svm_params = {"kernel": "rbf", "C": 10, "gamma": "scale", "random_state": 42}
        svm_model = SVC(**svm_params)
        svm_model.fit(X_train, y_train)
        y_pred_svm = svm_model.predict(X_test)

        acc_svm = accuracy_score(y_test, y_pred_svm)
        print("SVM Accuracy:", acc_svm)

        # Log parameters and metrics
        mlflow.log_params(svm_params)
        mlflow.log_metric("accuracy", acc_svm)

        # Log classification report
        report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
        report_path = os.path.join(save_path, "svm_classification_report.csv")
        pd.DataFrame(report_svm).transpose().to_csv(report_path)
        mlflow.log_artifact(report_path)

        # Log model itself
        mlflow.sklearn.log_model(svm_model, "svm_model")
        joblib.dump(svm_model, os.path.join(save_path, "svm_hand_gesture_model.pkl"))

    # ----------------- XGBOOST -----------------
    with mlflow.start_run(run_name="XGBoost"):
        xgb_params = {"use_label_encoder": False, "eval_metric": "mlogloss", "random_state": 42}
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)

        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        print("XGB Accuracy:", acc_xgb)

        # Log parameters and metrics
        mlflow.log_params(xgb_params)
        mlflow.log_metric("accuracy", acc_xgb)

        # Log classification report
        report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
        report_path = os.path.join(save_path, "xgb_classification_report.csv")
        pd.DataFrame(report_xgb).transpose().to_csv(report_path)
        mlflow.log_artifact(report_path)

    # Return the chosen SVM model and test set
    return svm_model, X_test, y_test

import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import argparse



def get_data():
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    #read the data s df
    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e


def evaluate(y_true, y_pred, pred_prob):
    # mae = mean_absolute_error(y_true, y_pred)
    # mse = mean_squared_error(y_true, y_pred)
    # rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # r2 = r2_score(y_true, y_pred)
    # return mae,mse,rmse,r2
    accuracy = accuracy_score(y_true,y_pred)
    roc_auc_scor = roc_auc_score(y_true,pred_prob,multi_class="ovr")

    return accuracy, roc_auc_scor


def main(n_estimator,max_depth):
    df=get_data()

    #Train - Test Split with raw data
    train, test = train_test_split(df)

    X_train=train.drop(["quality"],axis=1)
    X_test=test.drop("quality", axis=1)

    y_train=train[["quality"]]
    y_test=test[["quality"]]

    #Model Training
    """lr = ElasticNet()
    lr.fit(X_train,y_train)
    pred=lr.predict(X_test)"""

    with mlflow.start_run():
        rf=RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth)
        rf.fit(X_train,y_train)
        pred = rf.predict(X_test)
        pred_prob = rf.predict_proba(X_test)
        # #Evaluate the Model
        # mae, mse, rmse, r2 = evaluate(y_test,pred)
        # print(f"Mean Absolute Error {mae}\nmean squared error {mse}\nRoot Mean squared Error {rmse}\nr2 Score {r2}")

        accuracy, roc_auc_scor = evaluate(y_test,pred,pred_prob )
        mlflow.log_param("n_estimators", n_estimator)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("ROC AUC SCORE", roc_auc_scor)
        # print(f"Accuracy {accuracy}")
        # print(f"ROC AUC : {roc_auc_scor}")

        # mlflow model logging
        mlflow.sklearn.log_model(rf, "Random Forest Model")



if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--n_estimator", "-n", default=50, type=int)
    args.add_argument("--max_depth", "-m", default=5, type=int)
    parse_args = args.parse_args()

    try:
        main(n_estimator=parse_args.n_estimator, max_depth=parse_args.max_depth)
    except Exception as e:
        raise e









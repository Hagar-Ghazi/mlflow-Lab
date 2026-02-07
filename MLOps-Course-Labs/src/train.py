# """
# This module contains functions to preprocess and train the model
# for bank consumer churn prediction.
# """

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.utils import resample
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.compose import make_column_transformer
# from sklearn.preprocessing import OneHotEncoder,  StandardScaler
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
# )

# ### Import MLflow
# import mlflow
# import mlflow.sklearn
# import joblib

# def rebalance(data):
#     """
#     Resample data to keep balance between target classes.

#     The function uses the resample function to downsample the majority class to match the minority class.

#     Args:
#         data (pd.DataFrame): DataFrame

#     Returns:
#         pd.DataFrame): balanced DataFrame
#     """
#     churn_0 = data[data["Exited"] == 0]
#     churn_1 = data[data["Exited"] == 1]
#     if len(churn_0) > len(churn_1):
#         churn_maj = churn_0
#         churn_min = churn_1
#     else:
#         churn_maj = churn_1
#         churn_min = churn_0
#     churn_maj_downsample = resample(
#         churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
#     )

#     return pd.concat([churn_maj_downsample, churn_min])


# def preprocess(df):
#     """
#     Preprocess and split data into training and test sets.

#     Args:
#         df (pd.DataFrame): DataFrame with features and target variables

#     Returns:
#         ColumnTransformer: ColumnTransformer with scalers and encoders
#         pd.DataFrame: training set with transformed features
#         pd.DataFrame: test set with transformed features
#         pd.Series: training set target
#         pd.Series: test set target
#     """
#     filter_feat = [
#         "CreditScore",
#         "Geography",
#         "Gender",
#         "Age",
#         "Tenure",
#         "Balance",
#         "NumOfProducts",
#         "HasCrCard",
#         "IsActiveMember",
#         "EstimatedSalary",
#         "Exited",
#     ]
#     cat_cols = ["Geography", "Gender"]
#     num_cols = [
#         "CreditScore",
#         "Age",
#         "Tenure",
#         "Balance",
#         "NumOfProducts",
#         "HasCrCard",
#         "IsActiveMember",
#         "EstimatedSalary",
#     ]
#     data = df.loc[:, filter_feat]
#     data_bal = rebalance(data=data)
#     X = data_bal.drop("Exited", axis=1)
#     y = data_bal["Exited"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=1912
#     )
#     col_transf = make_column_transformer(
#         (StandardScaler(), num_cols), 
#         (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
#         remainder="passthrough",
#     )

#     X_train = col_transf.fit_transform(X_train)
#     X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

#     X_test = col_transf.transform(X_test)
#     X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

#     # Log the transformer as an artifact
#     joblib.dump(col_transf, "preprocessor.pkl")
#     mlflow.log_artifact("preprocessor.pkl")

#     return col_transf, X_train, X_test, y_train, y_test


# def train(X_train, y_train):
#     """
#     Train a logistic regression model.

#     Args:
#         X_train (pd.DataFrame): DataFrame with features
#         y_train (pd.Series): Series with target

#     Returns:
#         LogisticRegression: trained logistic regression model
#     """
#     log_reg = LogisticRegression(max_iter=1000)
#     log_reg.fit(X_train, y_train)

#     ### Log the model with the input and output schema
#     mlflow.sklearn.log_model(log_reg ,"model")

#     # Infer signature (input and output schema)
#     y_pred = log_reg.predict(X_train)
#     signature = mlflow.models.infer_signature(X_train, y_pred)

#     # Log model
#     mlflow.sklearn.log_model(
#     sk_model = log_reg, 
#     artifact_path = "model",  
#     signature = signature,   
#     input_example = X_train.head(5) 
# )
#     ### Log the data
#     churn_dataset = mlflow.data.from_pandas(X_train , name = "Churn Data")
#     mlflow.log_input(churn_dataset , context = "training_data")

#     return log_reg




# def main():
    

#     ### Set the tracking URI for MLflow
#     mlflow.set_tracking_uri("file:./mlruns")
#     mlflow.set_experiment("LR_churndata")

#     with mlflow.start_run(run_name = "Logistic Regression") as run:
#         ### Start a new run and leave all the main function code as part of the experiment
#         df = pd.read_csv(r"D:\projects_iti\mlflow_session\MLOps-Course-Labs\dataset\Churn_Modelling.csv")
#         col_transf, X_train, X_test, y_train, y_test = preprocess(df)

#         # Log parameters
#         max_iter = 1000
#         mlflow.log_param("max_iter", max_iter)
#         model = train(X_train, y_train)


#         ### Log metrics after calculating them
#         y_pred = model.predict(X_test)
#         mlflow.log_metrics({
#             "accuracy": accuracy_score(y_test, y_pred),
#             "precision": precision_score(y_test, y_pred),
#             "recall": recall_score(y_test, y_pred),
#             "f1_score": f1_score(y_test, y_pred)
#         })

#         ### Log tag
#         mlflow.set_tag("model_type", "LogisticRegression")


    
#     conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
#     conf_mat_disp = ConfusionMatrixDisplay(
#         confusion_matrix=conf_mat, display_labels=model.classes_
#     )
#     conf_mat_disp.plot()
    
#     # Log the image as an artifact in MLflow
#     plt.savefig("confusion_matrix.png")
#     mlflow.log_artifact("confusion_matrix.png")

#     plt.show()


# if __name__ == "__main__":
#     main()




################################### Three Models ##########################################

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

def rebalance(data):
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    churn_maj, churn_min = (churn_0, churn_1) if len(churn_0) > len(churn_1) else (churn_1, churn_0)
    
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )
    return pd.concat([churn_maj_downsample, churn_min])

def preprocess(df):
    filter_feat = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", 
                   "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
    cat_cols = ["Geography", "Gender"]
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
                "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)
    
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train_transformed = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train_transformed, columns=col_transf.get_feature_names_out())

    X_test_transformed = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test_transformed, columns=col_transf.get_feature_names_out())

    return col_transf, X_train, X_test, y_train, y_test

def train_and_log(model_obj, model_name, params, X_train, y_train, X_test, y_test):
    """
    Handles training, signing, and logging for a specific model instance.
    """
    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        # Log Parameters
        mlflow.log_params(params)
        mlflow.set_tag("model_type", type(model_obj).__name__)

        # Train
        model_obj.fit(X_train, y_train)

        # Metrics
        y_pred = model_obj.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)

        # Log Model with Signature
        signature = mlflow.models.infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model_obj, 
            artifact_path="model", 
            signature=signature,
            input_example=X_train.head(3)
        )

        # Confusion Matrix Artifact
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model_obj, X_test, y_test, ax=ax)
        plot_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close(fig)

        print(f"Finished logging {model_name} | F1: {metrics['f1_score']:.3f}")

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Bank_Churn_Comparison")

    # Load and Preprocess once
    df = pd.read_csv(r"D:\projects_iti\mlflow_session\MLOps-Course-Labs\dataset\Churn_Modelling.csv")
    col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    # Save preprocessor once as an experiment-level artifact (optional)
    joblib.dump(col_transf, "preprocessor.pkl")

    # Define the 3 models and their specific parameters
    model_configs = [
        {
            "name": "Logistic_Regression_Basic",
            "model": LogisticRegression(max_iter=1000, C=0.1),
            "params": {"max_iter": 1000, "C": 0.1, "solver": "lbfgs"}
        },
        {
            "name": "Random_Forest_Tuned",
            "model": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 10}
        },
        {
            "name": "SVM_RBF_Kernel",
            "model": SVC(kernel='rbf', C=1.0, probability=True),
            "params": {"kernel": "rbf", "C": 1.0}
        }
    ]

    # Iterate and train
    for config in model_configs:
        train_and_log(
            config["model"], 
            config["name"], 
            config["params"], 
            X_train, y_train, X_test, y_test
        )

if __name__ == "__main__":
    main()







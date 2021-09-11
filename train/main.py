from argparse import ArgumentParser
import dill
import pandas as pd
import numpy as np

from sklearn.datasets import load_wine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score


RANDOM_STATE = 99

class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, model, val_size=0.25, criterion="better_than_average"):
        assert criterion in ["better_than_average", "better_than_random"], \
            "Criterion must be either `better_than_average` or `better_than_random`"
        self.__metric = roc_auc_score
        self.__val_size = val_size
        self.__estimator = model 
        self.__original_features = []
        self.__feature_names = []

        self.__mean = np.mean
        self.__concat = pd.concat
        self.__DF = pd.DataFrame
        self.__Series = pd.Series

    @property
    def val_size(self):
        return self.__val_size

    def __train_val_split(self, X, y):
        df = self.__concat([X, y], axis=1)
        df_ = df.sample(frac=1, replace=False, random_state=RANDOM_STATE)

        val_length = int(df.shape[0] * self.__val_size)

        df_train = df_.iloc[val_length:, :]
        df_val = df_.iloc[:val_length, :]

        return (df_train.iloc[:, :-1], df_train.iloc[:, -1]), (df_val.iloc[:, :-1], df_val.iloc[:, -1])    
    
    def fit(self, X, y=None):
        assert isinstance(X, self.__DF), "Insert a pandas dataframe"
        assert isinstance(y, self.__Series), "Insert a pandas series"

        (X_train, y_train), (X_val, y_val) = self.__train_val_split(X, y)
        self.__original_features = X.columns
        metrics_results = {}

        for column in self.__original_features:
            input_data = X_train.loc[:, [column]]
            self.__estimator.fit(input_data, y_train)
            y_proba = self.__estimator.predict_proba(X_val.loc[:, [column]])
            metrics_results[column] = self.__metric(y_val, y_proba, multi_class="ovr")

        mean_performance = self.__mean(list(metrics_results.values()))
        self.__feature_names = [key for key, value in metrics_results.items() if value > mean_performance]
        features = "\n\t".join(self.__feature_names)
        print(f"Selected features:\n\t{features}")
        
        return self
    
    def transform(self, X):
        if not isinstance(X, self.__DF):
            X = self.__DF(X, columns=self.__original_features)
        return X.loc[:, self.__feature_names]

def get_raw_data():
    data = load_wine()
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="target")
    return X, y

def get_train_test_set(X, y, test_fraction=0.25):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=RANDOM_STATE)
    return (X_train, y_train), (X_test, y_test)

def get_model(val_size=0.25):
    model = Pipeline(steps=[
        ("feature_selector",
        FeatureSelector(
            model=GradientBoostingClassifier(n_estimators=15, max_depth=3, random_state=RANDOM_STATE),
            val_size=val_size
        )),
        ("model", GradientBoostingClassifier(n_estimators=15, max_depth=3, random_state=RANDOM_STATE))
    ])
    return model

def save_model(model, filename="wine.pkl"):
    with open("../app/models/"+filename, "wb") as file:
        dill.dump(model, file)

def print_training_results(model, test_set):
    X_test, y_test = test_set
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = ArgumentParser(description='Setting hyperparameters.')
    parser.add_argument('-v', '--val_size', required=True,
                        type=float, help='Fraction of dataset corresponding to validation set.')
    parser.add_argument('-t', '--test_size', required=True,
                        type=float, help='Fraction of dataset corresponding to test set.')
    args = parser.parse_args()

    X, y = get_raw_data()
    (X_train, y_train), (X_test, y_test) = get_train_test_set(X, y, test_fraction=args.test_size)
    model = get_model(val_size=args.val_size)
    model.fit(X_train, y_train)
    print_training_results(model, (X_test, y_test))
    save_model(model)

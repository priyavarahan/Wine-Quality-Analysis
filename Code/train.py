import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.regressors = [
            ('XGBoost Regression', XGBRegressor(objective='reg:squarederror'), {
                'n_estimators': [50],
                'max_depth': [3],
                'learning_rate': [0.1]
            }),
            ('Linear Regression', LinearRegression(), {}),
            ('Support Vector Regression', SVR(), {
                'C': [0.1],
                'kernel': ['linear']
            }),
            ('Random Forest Regression', RandomForestRegressor(), {
                'n_estimators': [50],
                'max_depth': [10]
            })
        ]
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}

    def train_models(self):
        for name, model, params in self.regressors:
            if params:
                grid_search = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(self.X, self.y)
                best_model = grid_search.best_estimator_
            else:
                best_model = model
                best_model.fit(self.X, self.y)

            predictions = best_model.predict(self.X)

            mae = mean_absolute_error(self.y, predictions) / 1e6
            r2 = r2_score(self.y, predictions)
            rmse = np.sqrt(mean_squared_error(self.y, predictions)) / 1e6

            self.metrics[name] = {
                'MAE': mae,
                'R2': r2,
                'RMSE': rmse
            }

        

            if self.best_model is None or rmse < self.metrics[self.best_model_name]['RMSE']:
                self.best_model = best_model
                self.best_model_name = name

    def get_best_model(self):
        return self.best_model

    def get_best_model_name(self):
        return self.best_model_name

    def get_metrics(self):
        return self.metrics

    def save_best_model(self, filename='best_model.pkl'):
        joblib.dump(self.best_model, filename)
        print(f"Best model saved as '{filename}'")

class DataHandler:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_csv(self.filename)

    def get_features_and_target(self):
        X = self.df[['YearsCodePro', 'CodingLanguageNum', 'OfficeStackAsyncNum', 'OpSysProfessionalNum', 'NEWCollabToolsNum', 'WorkExp', 'DatabaseNum', 'PlatformNum', 'OfficeStackSyncNum', 'Full-time Employment', 'Age_encoded', 'EdLevel_encoded', 'RemoteWork_encoded', 'Continent_encoded']]
        y = self.df['ConvertedCompYearly']
        return X, y

def main():
    file_name = 'Data/FinalEncodedWithCategorical.csv'

    data_handler = DataHandler(file_name)
    X, y = data_handler.get_features_and_target()

    model_trainer = ModelTrainer(X, y)
    model_trainer.train_models()
    best_model = model_trainer.get_best_model()
    best_model_name = model_trainer.get_best_model_name()
    metrics = model_trainer.get_metrics()

    model_trainer.save_best_model()

if __name__ == "__main__":
    main()

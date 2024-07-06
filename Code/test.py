import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

class DataHandler:
    def __init__(self, filename):
        self.filename = filename
        try:
            self.df = pd.read_csv(self.filename)
            print("CSV file loaded successfully.")
        except FileNotFoundError:
            print(f"File not found: {self.filename}")
            raise
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

    def get_features_and_target(self):
        X = self.df[['YearsCodePro', 'CodingLanguageNum', 'OfficeStackAsyncNum', 'OpSysProfessionalNum', 
                     'NEWCollabToolsNum', 'WorkExp', 'DatabaseNum', 'PlatformNum', 'OfficeStackSyncNum', 
                     'Full-time Employment', 'Age_encoded', 'EdLevel_encoded', 'RemoteWork_encoded', 
                     'Continent_encoded']]
        y = self.df['ConvertedCompYearly']
        return X, y

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.regressors = [
            ('XGBoost Regression', XGBRegressor(objective='reg:squarederror')),
            ('Linear Regression', LinearRegression()),
            ('Support Vector Regression', SVR()),
            ('Random Forest Regression', RandomForestRegressor())
        ]
        self.metrics = {}

    def train_models(self):
        for name, model in self.regressors:
            model.fit(self.X, self.y)
            predictions = model.predict(self.X)

            mae = mean_absolute_error(self.y, predictions)
            r2 = r2_score(self.y, predictions)
            rmse = np.sqrt(mean_squared_error(self.y, predictions))

            self.metrics[name] = {
                'MAE': mae,
                'R2': r2,
                'RMSE': rmse
            }

    def get_metrics(self):
        return self.metrics

    def read_metrics_from_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                
            metrics_data = []
            for line in lines:
                parts = line.strip().split(',')
                model_name = parts[0]
                rmse = float(parts[1])
                mae = float(parts[2])
                r2 = float(parts[3])
                metrics_data.append((model_name, rmse, mae, r2))
                
            return metrics_data
        except Exception as e:
            print(f"Error reading metrics from file: {e}")
            return []

    def display_metrics(self, metrics_data):
        if not metrics_data:
            print("No metrics data to display.")
            return

        df_metrics = pd.DataFrame(metrics_data, columns=['Model', 'RMSE', 'MAE', 'R2'])
        print("\nMetrics DataFrame:")
        print(df_metrics)

def main():
    file_name = 'Data/FinalEncodedWithCategorical.csv'
    filem = 'Data/_file.txt'

    # Initialize DataHandler
    data_handler = DataHandler(file_name)
    X, y = data_handler.get_features_and_target()

    
    model_trainer = ModelTrainer(X, y)
    model_trainer.train_models()

    metrics_data = model_trainer.read_metrics_from_file(filem)

   
    model_trainer.display_metrics(metrics_data)


if __name__ == "__main__":
    main()

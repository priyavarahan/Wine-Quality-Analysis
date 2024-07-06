import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import time

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.regressors = [
            ['Linear Regression', LinearRegression()],
            ['Support Vector Regression', SVR(C=0.1, kernel='linear')],
            ['Random Forest Regression', RandomForestRegressor(n_estimators=50)],
            ['XGBoost Regression', XGBRegressor(objective='reg:squarederror', n_estimators=50)]
        ]
        self.best_model = None

    def train_models(self):
        for name, model in self.regressors:
            model.fit(self.X_train, self.y_train)

            
            self.best_model = model

    def get_best_model(self):
        return self.best_model

class SalaryPredictor:
    def __init__(self, df, model):
        self.df = df
        self.model = model
        self.X = df[['YearsCodePro', 'CodingLanguageNum', 'OfficeStackAsyncNum', 'OpSysProfessionalNum', 'NEWCollabToolsNum', 'WorkExp', 'DatabaseNum', 'PlatformNum', 'OfficeStackSyncNum', 'Full-time Employment', 'Age_encoded', 'EdLevel_encoded', 'RemoteWork_encoded', 'Continent_encoded']]
        self.y = df['ConvertedCompYearly']
        
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def predict_salary(self, features_list):
        if len(features_list) == self.X_train.shape[1]:
            new_data = np.array([features_list])
            new_data = self.scaler.transform(new_data)
            predicted_salary = self.model.predict(new_data)
            return predicted_salary[0] + 20000  
        else:
            raise ValueError(f"Expected {self.X_train.shape[1]} features but received {len(features_list)}.")

    def train_and_predict(self):
        self.model.fit(self.X_train, self.y_train)
        return self.model

def main():
    
    file_name = 'Data/FinalEncodedWithCategorical.csv'
    if not os.path.exists(file_name):
        print(f"File '{file_name}' not found. Please check the file path and name.")
        return

    df = pd.read_csv(file_name)
    df = df.sample(n=1000, random_state=42)  
    
    model_trainer = ModelTrainer(df[['YearsCodePro', 'CodingLanguageNum', 'OfficeStackAsyncNum', 'OpSysProfessionalNum', 'NEWCollabToolsNum', 'WorkExp', 'DatabaseNum', 'PlatformNum', 'OfficeStackSyncNum', 'Full-time Employment', 'Age_encoded', 'EdLevel_encoded', 'RemoteWork_encoded', 'Continent_encoded']], df['ConvertedCompYearly'])
    model_trainer.train_models()
    best_model = model_trainer.get_best_model()

    
    salary_predictor = SalaryPredictor(df, best_model)

    # Interactive prompt for user input with detailed feature information
    prompt = (
        "-Enter the 14 features separated by commas in the following order:\n"
        "-Years of coding experience\n"
        "-Number of coding languages\n"
        "-Number of office stack\n"
        "-Number of operating system\n"
        "-Number of colab tools\n"
        "-Work experience\n"
        "-Number of database\n"
        "-Number of platform\n"
        "-Number of sync office stack\n"
        "-Full-time Employment (1 for Yes, 0 for No)\n"
        "-Age (0 for 18-24 years old, 1 for 25-34 years old, 2 for 35-44 years old, 3 for 45-54 years old, 4 for 55-64 years old)\n"
        "-EdLevel (0 for Associate degree, 1 for Bachelor’s degree, 2 for Master’s degree, 3 for Primary/elementary school, 4 for Professional degree, 5 for Secondary school, 6 for Some college/university study without earning a degree)\n"
        "-RemoteWork (0 for Hybrid, 1 for In-person, 2 for Remote)\n"
        "-Continent (0 for Africa, 1 for Asia, 2 for Europe, 3 for North America, 4 for Oceania, 5 for Others, 6 for South America)\n"
    )
    feature_input = input(prompt)
    features_list = [float(x) for x in feature_input.split(',')]
    try:
        predicted_salary = salary_predictor.predict_salary(features_list)
        print(f"Predicted Salary: ${predicted_salary:,.2f}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()

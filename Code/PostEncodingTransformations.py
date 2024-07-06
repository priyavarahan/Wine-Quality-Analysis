import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, shapiro
import math

class DataTransformer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.transformed_data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        print(self.data.head())

    def analyze_distribution(self, numeric_vars):
        num_vars = len(numeric_vars)
        num_cols = min(3, num_vars)
        num_rows = math.ceil(num_vars / num_cols)

        plt.figure(figsize=(5 * num_cols, 5 * num_rows))
        for i, var in enumerate(numeric_vars):
            plt.subplot(num_rows, num_cols, i + 1)
            sns.histplot(self.data[var], kde=True)
            plt.title(f'Distribution of {var}')
            plt.xlabel(var)
            plt.ylabel('Density')
        plt.tight_layout()
        plt.show()

        skewness = self.data[numeric_vars].apply(skew)
        print("Skewness:")
        for var, sk in skewness.items():
            print(f"{var}: {sk:.2f}")

        print("\nShapiro-Wilk Test for Normality:")
        for var in numeric_vars:
            stat, p = shapiro(self.data[var].dropna())
            if p > 0.05:
                result = "Normally distributed"
            else:
                result = "Not normally distributed"
            print(f"{var}: p-value = {p:.4f}, {result}")

    def detect_outliers(self, numeric_columns):
        outliers_dict = {}
        outliers_percentage_dict = {}

        for column in numeric_columns:
            outliers_indices = self.detect_outliers_iqr(column)
            outliers_dict[column] = outliers_indices
            total_rows = len(self.data)
            outliers_percentage = (len(outliers_indices) / total_rows) * 100 if total_rows > 0 else 0
            outliers_percentage_dict[column] = outliers_percentage

        for column, percentage in outliers_percentage_dict.items():
            print(f'Percentage of outliers in {column}: {percentage:.2f}%')

    def detect_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_indices = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)].index
        return outliers_indices

    def calculate_skewness(self, numerical_features):
        skewness = numerical_features.skew()
        print("Skewness for each feature:")
        print(skewness)

        logcomp_skewness = self.data['LogComp'].skew()
        print("\nSkewness of 'LogComp' (target feature):", logcomp_skewness)

    def calculate_outlier_percentage(self, data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        outlier_percentage = (len(outliers) / len(data)) * 100
        return outlier_percentage

    def plot_log_transformation(self, column):
        outlier_percentage_before = self.calculate_outlier_percentage(self.data, column)

        self.data[f'{column}_log'] = np.log1p(self.data[column])

        outlier_percentage_after = self.calculate_outlier_percentage(self.data, f'{column}_log')

        plt.figure(figsize=(16, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(self.data[column], kde=True)
        plt.title(f'Distribution of {column} (Before Log Transformation)\nOutliers: {outlier_percentage_before:.2f}%')
        plt.xlabel(column)
        plt.ylabel('Density')

        plt.subplot(1, 2, 2)
        sns.histplot(self.data[f'{column}_log'], kde=True)
        plt.title(f'Distribution of {column} (After Log Transformation)\nOutliers: {outlier_percentage_after:.2f}%')
        plt.xlabel(f'Log(1 + {column})')
        plt.ylabel('Density')

        plt.tight_layout()
        plt.show()

    def apply_log_transformations(self, columns_to_transform):
        for column in columns_to_transform:
            self.plot_log_transformation(column)

    def save_transformed_data(self, output_file):
        self.transformed_data = self.data.copy()
        self.transformed_data.to_csv(output_file, index=False)

# Usage
data_transformer = DataTransformer('Data/EncodedFinalCopy.csv')
data_transformer.load_data()

numeric_vars = ['Age_encoded', 'EdLevel_encoded', 'RemoteWork_encoded', 'YearsCodePro', 'Continent_encoded',
                'JobSatisfaction', 'CodingLanguageNum', 'OfficeStackAsyncNum', 'OpSysProfessionalNum', 'NEWCollabToolsNum',
                'WorkExp', 'DatabaseNum', 'PlatformNum', 'OfficeStackSyncNum', 'avg_score', 'ConvertedCompYearly',
                'LogComp', 'Full-time Employment']

data_transformer.analyze_distribution(numeric_vars)

numeric_columns = ['Age_encoded', 'EdLevel_encoded', 'RemoteWork_encoded', 'YearsCodePro', 'Continent_encoded',
                   'JobSatisfaction', 'CodingLanguageNum', 'OfficeStackAsyncNum', 'OpSysProfessionalNum', 'NEWCollabToolsNum',
                   'WorkExp', 'DatabaseNum', 'PlatformNum', 'OfficeStackSyncNum', 'avg_score', 'ConvertedCompYearly',
                   'LogComp', 'Full-time Employment']

data_transformer.detect_outliers(numeric_columns)

numerical_features = data_transformer.data.drop(['LogComp', 'ConvertedCompYearly'], axis=1)
data_transformer.calculate_skewness(numerical_features)

columns_to_transform = ['YearsCodePro', 'EdLevel_encoded', 'Continent_encoded', 'WorkExp', 'CodingLanguageNum',
                         'OfficeStackAsyncNum', 'OpSysProfessionalNum', 'NEWCollabToolsNum', 'DatabaseNum', 'PlatformNum',
                         'OfficeStackSyncNum']

data_transformer.apply_log_transformations(columns_to_transform)

data_transformer.save_transformed_data('readyformodeling.csv')

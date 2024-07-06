import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import kstest
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile


class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def explore_data(self):
        if self.data is not None:
            print("Info:")
            print(self.data.info())
            print("\nDescription:")
            print(self.data.describe())
            print("\nShape:")
            print(self.data.shape)
            print("\nData types of each column in the DataFrame:")
            for column in self.data.columns:
                print(f"{column}: {self.data[column].dtype}")
            print("\nNull value counts for each column:")
            print(self.data.isnull().sum())
        else:
            print("No data loaded. Use load_data() method first.")

    def drop_nulls_in_target(self, column):
        if self.data is not None:
            self.data = self.data.dropna(subset=[column])
        else:
            print("No data loaded. Use load_data() method first.")

    def detect_outliers_numeric(self, numeric_cols):
        if self.data is not None:
            for col in numeric_cols:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=self.data[col])
                plt.title(f'Boxplot of {col}')
                plt.show()
        else:
            print("No data loaded. Use load_data() method first.")

    def plot_numeric_distribution(self, numeric_vars):
        if self.data is not None:
            plt.figure(figsize=(12, 4))
            for i, var in enumerate(numeric_vars):
                plt.subplot(1, len(numeric_vars), i+1)
                sns.histplot(self.data[var], kde=True)
                plt.title(f'Distribution of {var}')
                plt.xlabel(var)
                plt.ylabel('Density')
                sm.qqplot(self.data[var], line='s', ax=plt.gca())
                plt.title(f'Q-Q Plot of {var}')
                kstest_result = kstest(self.data[var], 'norm')
                print(f"Kolmogorov-Smirnov test for {var}: p-value = {kstest_result.pvalue:.4f}")
            plt.tight_layout()
            plt.show()
        else:
            print("No data loaded. Use load_data() method first.")

    def calculate_outlier_percentage(self, column):
        if self.data is not None:
            Q1 = np.percentile(self.data[column], 25)
            Q3 = np.percentile(self.data[column], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
            outlier_percentage = (len(outliers) / len(self.data)) * 100
            return outlier_percentage
        else:
            print("No data loaded. Use load_data() method first.")

    def plot_salary_by_experience(self):
        if self.data is not None:
            plt.figure(figsize=(20, 6))
            sns.scatterplot(x='YearsCodePro', y='ConvertedCompYearly', data=self.data)
            plt.title('Salary by Experience')
            plt.xlabel('Years of Coding Experience')
            plt.ylabel('Converted Compensation (Yearly)')
            plt.show()
        else:
            print("No data loaded. Use load_data() method first.")

    def plot_remote_work_pie_chart(self):
        if self.data is not None:
            remote_work_counts = self.data['RemoteWork'].value_counts()
            plt.figure(figsize=(8, 6))
            plt.pie(remote_work_counts, labels=remote_work_counts.index, autopct='%1.1f%%', startangle=140)
            plt.title('Distribution of Remote Work')
            plt.axis('equal')
            plt.show()
        else:
            print("No data loaded. Use load_data() method first.")


# Example usage:
compress_file_path = 'Data/raw_data.csv.zip'
file_path = 'Data/'
with ZipFile(compress_file_path, 'r') as zObject: 
    zObject.extractall(path= file_path)

data_path = file_path+'raw_data.csv'
processor = DataProcessor(data_path)
processor.load_data()
processor.explore_data()
processor.drop_nulls_in_target('ConvertedCompYearly')
processor.detect_outliers_numeric(['YearsCodePro', 'ConvertedCompYearly'])
processor.plot_numeric_distribution(['ConvertedCompYearly'])
outlier_percentage = processor.calculate_outlier_percentage('ConvertedCompYearly')
print(f'Percentage of outliers in ConvertedCompYearly: {outlier_percentage:.2f}%')
processor.plot_salary_by_experience()
processor.plot_remote_work_pie_chart()

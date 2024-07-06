import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import ggplot, aes, geom_point, geom_smooth, labs, theme
from tabulate import tabulate
from zipfile import ZipFile 
# importing the zipfile module 


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.load_data()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def display_data(self):
        print(self.data)

    def transform_target_feature(self):
        self.data['LogComp'] = np.log(self.data['ConvertedCompYearly'])

    def plot_log_transformation(self, column):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        sns.histplot(self.data[column], kde=True, ax=axs[0])
        axs[0].set_title(f'Distribution of {column} (Before Log Transformation)')
        axs[0].set_xlabel(column)
        axs[0].set_ylabel('Density')

        sns.histplot(self.data['LogComp'], kde=True, ax=axs[1])
        axs[1].set_title(f'Distribution of Log Transformed {column}')
        axs[1].set_xlabel('Log(1 + ConvertedCompYearly)')
        axs[1].set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    def plot_compensation_vs_experience(self, column, color_column):
        plot = (
            ggplot(self.data, aes(x=column, y='LogComp', color=color_column)) +
            geom_point() +
            geom_smooth(method='lm', se=False) +
            labs(title=f'Yearly Compensation vs {column.replace("_", " ").title()}',
                 x=column.replace("_", " ").title(),
                 y='Log Converted Yearly Compensation',
                 color=color_column.replace("_", " ").title()) +
            theme(figure_size=(16, 6))
        )
        print(plot)

    def plot_satisfaction_vs_salary(self, salary_threshold):
        filtered_data = self.data.dropna(subset=['ConvertedCompYearly', 'JobSatisfaction', 'Age'])

        scatter_plot = (
            ggplot(filtered_data[filtered_data['ConvertedCompYearly'] <= salary_threshold],
                   aes(x='ConvertedCompYearly', y='JobSatisfaction', color='factor(Age)')) +
            geom_point() +
            labs(x='Yearly Salary (USD)', y='Job Satisfaction', color='Age') +
            theme(figure_size=(10, 6))
        )

        print(scatter_plot)

    def plot_job_satisfaction_by_org_size(self):
        sns.set_style("dark")

        plt.figure(figsize=(12, 8))
        ax = sns.countplot(data=self.data, x='JobSatisfaction', hue='OrgSize', palette='Set2')

        plt.xlabel('Job Satisfaction', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title('Distribution of Job Satisfaction by Organization Size', fontsize=16)
        plt.legend(title='Organization Size', fontsize=12, title_fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.tight_layout()
        plt.show()

    def print_question_responses(self):
        data = {
            "Questions": [
                "1) I have interactions with people outside of my immediate team.",
                "2) Knowledge silos prevent me from getting ideas across the organization (i.e., one individual or team has information that isn't shared with others).",
                "3) I can find up-to-date information within my organization to help me do my job.",
                "4) I am able to quickly find answers to my questions with existing tools and resources.",
                "5) I know which system or resource to use to find information and answers to questions I have.",
                "6) I often find myself answering questions that I've already answered before.",
                "7) Waiting on answers to questions often causes interruptions and disrupts my workflow.",
                "8) I feel like I have the tools and/or resources to quickly understand and work on any area of my company's code/system/platform."
            ],
            "Strongly disagree": [1, 5, 1, 1, 1, 5, 5, 1],
            "Disagree": [2, 4, 2, 2, 2, 4, 4, 2],
            "Neither agree nor disagree": [3, 3, 3, 3, 3, 3, 3, 3],
            "Agree": [4, 2, 4, 4, 4, 2, 2, 4],
            "Strongly agree": [5, 1, 5, 5, 5, 1, 1, 5]
        }

        df = pd.DataFrame(data)
        print(tabulate(df, headers='keys', tablefmt='pretty'))

    def create_count_columns(self):
        self.data['CodingLanguageNum'] = self.data['LanguageHaveWorkedWith'].str.count(';') + 1
        self.data.loc[self.data['LanguageHaveWorkedWith'].isna() | (self.data['LanguageHaveWorkedWith'] == ""), 'CodingLanguageNum'] = 0

        self.data['DatabaseNum'] = self.data['DatabaseHaveWorkedWith'].str.count(';') + 1
        self.data.loc[self.data['DatabaseHaveWorkedWith'].isna() | (self.data['DatabaseHaveWorkedWith'] == ""), 'DatabaseNum'] = 0

        self.data['PlatformNum'] = self.data['PlatformHaveWorkedWith'].str.count(';') + 1
        self.data.loc[self.data['PlatformHaveWorkedWith'].isna() | (self.data['PlatformHaveWorkedWith'] == ""), 'PlatformNum'] = 0

        self.data['WebframeNum'] = self.data['WebframeHaveWorkedWith'].str.count(';') + 1
        self.data.loc[self.data['WebframeHaveWorkedWith'].isna() | (self.data['WebframeHaveWorkedWith'] == ""), 'WebframeNum'] = 0

        self.data['ToolsTechNum'] = self.data['ToolsTechHaveWorkedWith'].str.count(';') + 1
        self.data.loc[self.data['ToolsTechHaveWorkedWith'].isna() | (self.data['ToolsTechHaveWorkedWith'] == ""), 'ToolsTechNum'] = 0

        self.data['NEWCollabToolsNum'] = self.data['NEWCollabToolsHaveWorkedWith'].str.count(';') + 1
        self.data.loc[self.data['NEWCollabToolsHaveWorkedWith'].isna() | (self.data['NEWCollabToolsHaveWorkedWith'] == ""), 'NEWCollabToolsNum'] = 0

        self.data['OfficeStackAsyncNum'] = self.data['OfficeStackAsyncHaveWorkedWith'].str.count(';') + 1
        self.data.loc[self.data['OfficeStackAsyncHaveWorkedWith'].isna() | (self.data['OfficeStackAsyncHaveWorkedWith'] == ""), 'OfficeStackAsyncNum'] = 0

        self.data['OfficeStackSyncNum'] = self.data['OfficeStackSyncHaveWorkedWith'].str.count(';') + 1
        self.data.loc[self.data['OfficeStackSyncHaveWorkedWith'].isna() | (self.data['OfficeStackSyncHaveWorkedWith'] == ""), 'OfficeStackSyncNum'] = 0

        self.data['OpSysProfessionalNum'] = self.data['OpSysProfessional use'].str.count(';') + 1
        self.data.loc[self.data['OpSysProfessional use'].isna() | (self.data['OpSysProfessional use'] == ""), 'OpSysProfessionalNum'] = 0

    def drop_original_columns(self):
        self.data.drop(['LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith', 'PlatformHaveWorkedWith', 'WebframeHaveWorkedWith', 'ToolsTechHaveWorkedWith', 'NEWCollabToolsHaveWorkedWith', 'OfficeStackAsyncHaveWorkedWith', 'OfficeStackSyncHaveWorkedWith', 'OpSysProfessional use'], axis=1, inplace=True)

    def plot_compensation_vs_language_count(self, color_column):
        plot = (
            ggplot(self.data, aes(x='CodingLanguageNum', y='ConvertedCompYearly', color=color_column)) +
            geom_point() +
            geom_smooth(method='lm', se=False) +
            labs(title='Yearly Compensation vs Number of Coding Languages',
                 x='Number of Coding Languages',
                 y='Converted Yearly Compensation',
                 color=color_column.replace("_", " ").title()) +
            theme(figure_size=(16, 6))
        )
        print(plot)

    def plot_compensation_vs_language_count_log(self, color_column):
        plot = (
            ggplot(self.data, aes(x='CodingLanguageNum', y='LogComp', color=color_column)) +
            geom_point() +
            geom_smooth(method='lm', se=False) +
            labs(title='Yearly Compensation vs Number of Coding Languages',
                 x='Number of Coding Languages',
                 y='Converted Yearly Compensation',
                 color=color_column.replace("_", " ").title()) +
            theme(figure_size=(16, 6))
        )
        print(plot)

    def plot_language_count_distribution(self):
        plt.hist(self.data['CodingLanguageNum'], bins='auto', color='skyblue', edgecolor='black')
        plt.xlabel('Number of Languages')
        plt.ylabel('Frequency')
        plt.title('Distribution of Number of Coding Languages Known')
        plt.show()

    def print_language_count_frequency(self):
        language_counts = self.data['CodingLanguageNum'].value_counts().sort_index()
        print(language_counts)

        plt.bar(language_counts.index, language_counts.values, color='lightblue', edgecolor='black')
        plt.title("Number of Individuals vs. Number of Languages Known")
        plt.xlabel("Number of Languages Known")
        plt.ylabel("Number of Individuals")
        plt.show()

    def explore_column(self, column):
        print("Exploring column:", column)

        if self.data[column].dtype in ['int64', 'float64']:
            print("Type: Numeric")
            print("Summary statistics:")
            print(self.data[column].describe())
        else:
            print("Type: Categorical")
            print("Unique values/levels:")
            print(self.data[column].value_counts())
        print()

    def explore_data(self):
        print("Missing values per column:")
        print(self.data.isnull().sum())
        print("\nData shape:", self.data.shape)

        print("\nUnique values in 'EdLevel':")
        print(self.data['EdLevel'].unique())

        print("\nUnique values in 'OrgSize':")
        print(self.data['OrgSize'].unique())

    def filter_data(self):
        self.data = self.data[~self.data['EdLevel'].isin(['Something else'])]
        self.data = self.data[~self.data['OrgSize'].isin(['Just me - I am a freelancer, sole proprietor, etc.', 'I don\'t know'])]

    def print_unique_values(self):
        categorical_columns = ['Age', 'RemoteWork', 'EdLevel', 'OrgSize', 'AISelect', 'JobSatisfaction', 'WorkExp', 'YearsCodePro']
        for column in categorical_columns:
            print(f"Unique values for '{column}':", self.data[column].unique())

    def handle_yearscodepro(self):
        self.data['YearsCodePro'].replace({"Less than 1 year": 0.5, "More than 50 years": 51}, inplace=True)

    def drop_age_rows(self):
        self.data = self.data[~self.data['Age'].isin(["Under 18 years old", "Prefer not to say"])]

    def preprocess_data(self):
        nominal_attributes = ['RemoteWork', 'Employment', 'Age', 'CodingActivities', 'DevType', 'JobSatisfaction', 'LearnCode', 'LearnCodeOnline', 'BuyNewTool']
        self.data[nominal_attributes] = self.data[nominal_attributes].fillna(self.data[nominal_attributes].mode().iloc[0])

        ordinal_mappings = {
            'OrgSize': {'10,000 or more employees': 1,
                        '5,000 to 9,999 employees': 2,
                        '1,000 to 4,999 employees': 3,
                        '500 to 999 employees': 4,
                        '100 to 499 employees': 5,
                        '20 to 99 employees': 6,
                        '10 to 19 employees': 7,
                        '2 to 9 employees': 8},
            'PurchaseInfluence': {'I have a great deal of influence': 1, 'I have some influence': 2, 'I have little or no influence': 3},
            'SOVisitFreq': {'Multiple times per day': 1, 'Daily or almost daily': 2, 'A few times per week': 3,
                            'A few times per month or weekly': 4, 'Less than once per month or monthly': 5},
            'SOAccount': {'Yes': 1, 'No': 2, "Not sure/can't remember": 3},
            'SOPartFreq': {'Multiple times per day': 1, 'Daily or almost daily': 2, 'A few times per week': 3,
                           'A few times per month or weekly': 4, 'Less than once per month or monthly': 5,
                           'I have never participated in Q&A on Stack Overflow': 6},
            'SOComm': {'Yes, definitely': 1, 'Yes, somewhat': 2, 'Neutral': 3, 'No, not really': 4,
                       'No, not at all': 5, 'Not sure': 6},
            'SurveyLength': {'Too short': 1, 'Appropriate in length': 2, 'Too long': 3},
            'SurveyEase': {'Easy': 1, 'Neither easy nor difficult': 2, 'Difficult': 3},
        }

        for column, mapping in ordinal_mappings.items():
            if column in ['OrgSize', 'PurchaseInfluence', 'SOVisitFreq', 'SOAccount', 'SOPartFreq', 'SOComm', 'SurveyLength', 'SurveyEase']:
                self.data[column] = self.data[column].map(mapping)
                self.data[column].fillna(self.data[column].median(), inplace=True)
                self.data['YearsCodePro'] = pd.to_numeric(self.data['YearsCodePro'], errors='coerce')
                self.data['YearsCodePro'] = self.data['YearsCodePro'].fillna(self.data['YearsCodePro'].mean())

                median_workexp = self.data['WorkExp'].median()
                self.data['WorkExp'].fillna(median_workexp, inplace=True)

    def check_unique_values(self):
        columns_to_check = ['OrgSize', 'PurchaseInfluence', 'SOVisitFreq', 'SOAccount', 'SOPartFreq', 'SOComm', 'SurveyLength', 'SurveyEase']
        for column in columns_to_check:
            unique_values = self.data[column].unique()
            print(f"Unique values in '{column}': {unique_values}")

    def drop_columns(self):
        columns_to_drop = ['Knowledge_1', 'Knowledge_2', 'Knowledge_3', 'Knowledge_4',
                           'Knowledge_5', 'Knowledge_6', 'Knowledge_7', 'Knowledge_8',
                           'Knowledge_1_num', 'Knowledge_2_num', 'Knowledge_3_num', 'Knowledge_4_num',
                           'Knowledge_5_num', 'Knowledge_6_num', 'Knowledge_7_num', 'Knowledge_8_num']
        self.data.drop(columns=columns_to_drop, inplace=True)

    def save_data(self, output_file):
        self.data.to_csv(output_file, index=False)

if __name__ == "__main__":
    compress_file_path = 'Data/new_data.csv.zip'
    file_path = 'Data/'
    with ZipFile(compress_file_path, 'r') as zObject: 
        zObject.extractall(path= file_path)

    data_preprocessor = DataPreprocessor(file_path+'new_data.csv')

    data_preprocessor.display_data()
    data_preprocessor.transform_target_feature()
    data_preprocessor.plot_log_transformation('ConvertedCompYearly')
    data_preprocessor.plot_compensation_vs_experience('WorkExp', 'Age')
    data_preprocessor.plot_satisfaction_vs_salary(600000)
    data_preprocessor.plot_job_satisfaction_by_org_size()
    data_preprocessor.print_question_responses()
    data_preprocessor.create_count_columns()
    data_preprocessor.drop_original_columns()
    data_preprocessor.plot_compensation_vs_language_count('EdLevel')
    data_preprocessor.plot_compensation_vs_language_count_log('RemoteWork')
    data_preprocessor.plot_compensation_vs_language_count_log('Age')
    data_preprocessor.plot_language_count_distribution()
    data_preprocessor.print_language_count_frequency()
    data_preprocessor.explore_data()
    data_preprocessor.filter_data()
    data_preprocessor.print_unique_values()
    data_preprocessor.handle_yearscodepro()
    data_preprocessor.drop_age_rows()
    data_preprocessor.preprocess_data()
    data_preprocessor.check_unique_values()
    data_preprocessor.drop_columns()
    data_preprocessor.save_data('NullImputedFinal.csv')

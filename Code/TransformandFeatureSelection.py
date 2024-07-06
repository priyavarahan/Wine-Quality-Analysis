import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from tabulate import tabulate

class DataTransformer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        print("Data loaded successfully.")

    def remove_salary_outliers(self):
        salary_threshold = self.data['ConvertedCompYearly'].quantile(0.934)
        self.filtered_data = self.data[self.data['ConvertedCompYearly'] <= salary_threshold]
        print("Outliers removed from salary data.")

    def plot_salary_histogram(self):
        breaks = range(0, int(self.filtered_data['ConvertedCompYearly'].max()) + 50000, 50000)
        labels = [f'{x/1000:.0f}k' for x in breaks]

        plt.hist(self.filtered_data['ConvertedCompYearly'], bins=breaks, color='lightblue', edgecolor='black')
        plt.xlabel('Yearly Salary (USD)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Yearly Salaries (Excluding 4.59% outliers)')
        plt.xticks(breaks, labels)
        plt.show()
        print("Salary histogram plotted.")

    def calculate_outlier_percentage(self, column):
        Q1 = np.percentile(self.filtered_data[column], 25)
        Q3 = np.percentile(self.filtered_data[column], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.filtered_data[(self.filtered_data[column] < lower_bound) | (self.filtered_data[column] > upper_bound)]
        outlier_percentage = (len(outliers) / len(self.filtered_data)) * 100
        print(f"Outlier percentage in {column} after filtering: {outlier_percentage:.2f}%")
        return outlier_percentage

    def feature_selection_anova(self):
        categorical_features = self.data[['Q120', 'EdLevel', 'MainBranch', 'Age', 'Employment', 'RemoteWork', 'SOComm', 'SOAccount', 'CodingActivities', 'SOVisitFreq', 'SOPartFreq', 'EdLevel', 'OrgSize', 'PurchaseInfluence', 'LearnCode', 'LearnCodeOnline', 'YearsCodePro', 'DevType', 'BuyNewTool', 'Country', 'AISelect', 'TBranch', 'JobSatisfaction', 'SurveyLength', 'SurveyEase']]
        anova_results = []

        for column in categorical_features.columns:
            if len(self.data[column].unique()) > 1:
                anova_result = f_oneway(*[group["ConvertedCompYearly"] for name, group in self.data.groupby(column)])
                anova_results.append({"Feature": column, "F-value": anova_result.statistic, "P-value": anova_result.pvalue})

        self.anova_df = pd.DataFrame(anova_results)
        print("ANOVA analysis for feature selection completed.")

    def select_significant_features(self, threshold=0.05):
        self.significant_features = self.anova_df[(self.anova_df['P-value'] < threshold) | (self.anova_df['Feature'].isin(['AISelect', 'JobSatisfaction']))]
        self.significant_feature_names = self.significant_features['Feature'].tolist()
        self.data_filtered = self.data[self.significant_feature_names]
        print("Significant features selected based on ANOVA analysis.")

    def correlation_analysis(self):
        numeric_features = ['CodingLanguageNum', 'DatabaseNum', 'PlatformNum', 'WebframeNum', 'ToolsTechNum', 'NEWCollabToolsNum', 'OfficeStackAsyncNum', 'OfficeStackSyncNum', 'OpSysProfessionalNum', 'YearsCodePro', 'ResponseId', 'WorkExp', 'ConvertedCompYearly', 'avg_score']
        numeric_data = self.data[numeric_features]
        self.correlation_matrix = numeric_data.corr()
        self.correlation_with_target = self.correlation_matrix['ConvertedCompYearly'].drop('ConvertedCompYearly')
        print("Correlation analysis completed.")

    def plot_correlation_matrix(self):
        numeric_features = ['CodingLanguageNum', 'DatabaseNum', 'PlatformNum', 'WebframeNum', 'ToolsTechNum', 'NEWCollabToolsNum', 'OfficeStackAsyncNum', 'OfficeStackSyncNum', 'OpSysProfessionalNum', 'YearsCodePro', 'ResponseId', 'WorkExp', 'ConvertedCompYearly', 'avg_score']
        numeric_data = self.data[numeric_features]
        correlation_matrix = numeric_data.corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title("Pearson's Correlation Matrix")
        plt.show()
        print("Correlation matrix plotted.")

    def select_columns(self):
        self.selected_columns = ['Age', 'Employment', 'RemoteWork', 'EdLevel', 'YearsCodePro', 'Country', 'JobSatisfaction', 'CodingLanguageNum', 'OfficeStackAsyncNum', 'OpSysProfessionalNum', 'NEWCollabToolsNum', 'WorkExp', 'DatabaseNum', 'PlatformNum', 'OfficeStackSyncNum', 'avg_score', 'ConvertedCompYearly', 'LogComp']
        self.selected_data = self.data[self.selected_columns]
        print("Selected columns for analysis.")

    def create_summary_table(self):
        clean_column_names = {
            'Age': 'Age',
            'Employment': 'Employment',
            'RemoteWork': 'Remote Work',
            'EdLevel': 'Education Level',
            'YearsCodePro': 'Years of Professional Coding Experience',
            'Country': 'Country',
            'JobSatisfaction': 'Job Satisfaction',
            'CodingLanguageNum': 'Number of Coding Languages',
            'OfficeStackAsyncNum': 'Number of Asynchronous Office Tools',
            'OpSysProfessionalNum': 'Number of Operating Systems Used Professionally',
            'NEWCollabToolsNum': 'Number of Collaboration Tools Used',
            'WorkExp': 'Work Experience',
            'DatabaseNum': 'Number of Databases Used',
            'PlatformNum': 'Number of Platforms Used',
            'OfficeStackSyncNum': 'Number of Synchronous Office Tools',
            'avg_score': 'Average Score',
            'LogComp': 'Converted Yearly Compensation'
        }

        self.table_data_clean = self.selected_data.rename(columns=clean_column_names)
        selected_columns = ['Age', 'Employment', 'Remote Work', 'Education Level', 'Years of Professional Coding Experience', 'Country', 'Job Satisfaction', 'Number of Coding Languages', 'Number of Asynchronous Office Tools', 'Number of Operating Systems Used Professionally', 'Number of Collaboration Tools Used', 'Work Experience', 'Number of Databases Used', 'Number of Platforms Used', 'Number of Synchronous Office Tools', 'Average Score', 'Converted Yearly Compensation']
        summary_table = self.table_data_clean[selected_columns].describe().transpose()
        print(tabulate(summary_table, headers='keys', tablefmt='pipe'))
        print("Summary table created.")

    def get_country_value_counts(self):
        self.country_value_counts = self.data_filtered['Country'].value_counts()
        print("Country value counts:")
        print(self.country_value_counts)

    def map_countries_to_continents(self):
        continent_mapping = {'North America': ['United States of America', 'Canada', 'Mexico', 'Costa Rica', 'Honduras'],
                            'South America': ['Brazil', 'Argentina', 'Colombia', 'Peru', 'Venezuela'],
                                'Europe': ['Germany', 'United Kingdom of Great Britain and Northern Ireland', 'France', 'Italy', 'Spain', 'Poland',
'Netherlands', 'Sweden', 'Switzerland', 'Portugal', 'Czech Republic', 'Austria', 'Denmark', 'Norway',
'Finland', 'Romania', 'Belgium', 'Russian Federation', 'Turkey', 'Ukraine', 'Greece', 'Hungary',
'Ireland', 'Serbia', 'Bulgaria', 'Slovakia', 'Lithuania', 'Slovenia', 'Croatia', 'Estonia', 'Latvia',
'Luxembourg', 'Bosnia and Herzegovina', 'Republic of Moldova', 'Montenegro', 'Albania', 'Kosovo', 'North Macedonia'],
'Asia': ['India', 'Japan', 'China', 'South Korea', 'Indonesia', 'Malaysia', 'Pakistan', 'Bangladesh', 'Philippines',
'Viet Nam', 'Singapore', 'Thailand', 'Hong Kong (S.A.R.)', 'Taiwan', 'United Arab Emirates', 'Israel', 'Iran',
'Saudi Arabia', 'Iraq', 'Kazakhstan', 'Uzbekistan', 'Qatar', 'Lebanon', 'Oman', 'Jordan', 'Syrian Arab Republic',
'Yemen', 'State of Palestine', 'Afghanistan', 'Sri Lanka', 'Nepal', 'Maldives', 'Kuwait', 'Georgia', 'Azerbaijan',
'Armenia', 'Bahrain', 'Timor-Leste', 'Cyprus', 'Bhutan', 'Republic of Korea'],
'Africa': ['Nigeria', 'South Africa', 'Egypt', 'Kenya', 'Ethiopia', 'Morocco', 'Ghana', 'Uganda', 'Algeria',
'Sudan', 'Tanzania', 'Angola', 'Zimbabwe', 'Tunisia', 'Zambia', 'Rwanda', 'Mauritius', 'Mozambique',
'Cameroon', 'Ivory Coast', 'Madagascar', 'Senegal', 'Malawi', 'Uzbekistan', 'Senegal', 'Chad', 'Togo',
'Sierra Leone', 'Burkina Faso', 'Mali', 'Mauritania', 'Eritrea', 'Namibia', 'Gambia', 'Botswana', 'Gabon',
'Lesotho', 'Guinea-Bissau', 'Equatorial Guinea', 'Seychelles', 'Cabo Verde', 'Comoros', 'São Tomé and Príncipe'],
'Oceania': ['Australia', 'New Zealand', 'Fiji', 'Papua New Guinea', 'Solomon Islands', 'Vanuatu', 'Samoa', 'Kiribati',
'Tonga', 'Micronesia', 'Palau', 'Marshall Islands', 'Tuvalu', 'Nauru'],
'Other': ['Other']}
        
        def map_to_continent(country):
            for continent, countries in continent_mapping.items():
                if country in countries:
                    return continent
            return 'Other'

        self.selected_data['Continent'] = self.selected_data['Country'].apply(map_to_continent)
        self.selected_data.drop(columns=['Country'], inplace=True)
        print("Countries mapped to continents.")

    def get_continent_value_counts(self):
        self.continent_counts = self.selected_data['Continent'].value_counts()
        print("Continent value counts:")
        print(self.continent_counts)

    def save_data(self, output_file):
        self.selected_data.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

if __name__ == "__main__":
    data_path = 'Data/NullImputedFinal.csv'
    data_transformer = DataTransformer(data_path)

    # Remove salary outliers
    data_transformer.remove_salary_outliers()

    # Plot salary histogram
    data_transformer.plot_salary_histogram()

    # Calculate outlier percentage for ConvertedCompYearly
    outlier_percentage = data_transformer.calculate_outlier_percentage('ConvertedCompYearly')
    print(f'Percentage of outliers in ConvertedCompYearly after filtering: {outlier_percentage:.2f}%')

    # Feature selection using ANOVA
    data_transformer.feature_selection_anova()
    data_transformer.select_significant_features()

    # Correlation analysis
    data_transformer.correlation_analysis()
    print("Correlation with target variable:")
    print(data_transformer.correlation_with_target)

    # Plot correlation matrix
    data_transformer.plot_correlation_matrix()

    # Select columns
    data_transformer.select_columns()

    # Create summary table
    data_transformer.create_summary_table()

    # Get country value counts
    data_transformer.get_country_value_counts()

    # Map countries to continents
    data_transformer.map_countries_to_continents()
    data_transformer.get_continent_value_counts()

    # Save data to a new CSV file
    data_transformer.save_data('visualisation.csv')

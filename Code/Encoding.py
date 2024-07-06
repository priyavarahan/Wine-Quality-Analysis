import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataEncoder:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.encoded_data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        print("Loaded data:")
        print(self.data.head())

    def encode_categories(self):
        # Split the 'Employment' column by delimiter and create separate columns
        split_employment = self.data['Employment'].str.get_dummies(';')
        split_employment = split_employment * 1

        # Concatenate the one-hot encoded columns with the original DataFrame
        self.data = pd.concat([self.data, split_employment], axis=1)

        # Drop the original 'Employment' column
        self.data.drop('Employment', axis=1, inplace=True)

        # Rename the columns
        self.data = self.data.rename(columns={
            'Employed, full-time': 'Full-time Employment',
            'Employed, part-time': 'Part-time Employment',
            'I prefer not to say': 'Prefer not to say',
            'Independent contractor, freelancer, or self-employed': 'Self-Employed',
            'Retired': 'Retired'
        })

        # Delete rows where age is 65 years or older
        self.data = self.data[self.data['Age'] != '65 years or older']

        # Drop specified columns
        self.data.drop(['Part-time Employment', 'Prefer not to say', 'Self-Employed', 'Retired'], axis=1, inplace=True)

        # Label encode 'Age', 'EdLevel', 'RemoteWork', and 'Continent'
        age_label_encoder = LabelEncoder()
        edlevel_label_encoder = LabelEncoder()
        remotework_encoder = LabelEncoder()
        continent_label_encoder = LabelEncoder()

        self.data['Age_encoded'] = age_label_encoder.fit_transform(self.data['Age'])
        self.data['EdLevel_encoded'] = edlevel_label_encoder.fit_transform(self.data['EdLevel'])
        self.data['RemoteWork_encoded'] = remotework_encoder.fit_transform(self.data['RemoteWork'])
        self.data['Continent_encoded'] = continent_label_encoder.fit_transform(self.data['Continent'])

        # Encoding 'JobSatisfaction' using label encoding
        self.data['JobSatisfaction'] = self.data['JobSatisfaction'].map({'Neutral': 0, 'Satisfied': 1, 'Not satisfied': 2})

        # Drop specified columns after encoding
        self.data.drop(columns=["Continent", "Age", "RemoteWork", "EdLevel"], inplace=True)

        self.encoded_data = self.data.copy()
        print("\nEncoded data summary:")
        print(self.encoded_data.head())

    def save_encoded_data(self, output_file):
        self.encoded_data.to_csv(output_file, index=False)
        print(f"\nEncoded data saved to {output_file}")

# Usage
data_encoder = DataEncoder('Data/visualisation.csv')
data_encoder.load_data()
data_encoder.encode_categories()
data_encoder.save_encoded_data('FinalEncodedWithCategorical.csv')

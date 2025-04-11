import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# Load dataset
file_path = 'PS215192-569836982-dum.xls'

# Read the Excel file
data = pd.read_excel(file_path, sheet_name='Subject Data')

# top factors selected using heap map

columns_to_select = [
    # 'PharmGKB Subject ID',
    # 'PharmGKB Sample ID',
    'Gender',
    'Age',
    'Height (cm)',
    'Weight (kg)',
    'Indication for Warfarin Treatment',
    'Aspirin',
    'Simvastatin (Zocor)',
    'Amiodarone (Cordarone)',
    'Target INR',
    'Therapeutic Dose of Warfarin',
    'INR on Reported Therapeutic Dose of Warfarin',
    'Current Smoker',
    'Cyp2C9 genotypes',
    'CYP2C9 consensus',
    'VKORC1 genotype:   -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
    'VKORC1 genotype:   3730 G>A (9041); chr16:31009822; rs7294;  A/G',
    'VKORC1 genotype:   1542G>C (6853); chr16:31012010; rs8050894; C/G',
    'VKORC1 genotype:   1173 C>T(6484); chr16:31012379; rs9934438; A/G',
    'VKORC1 1173 consensus',
    'VKORC1 2255 consensus',
    'VKORC1 genotype:   2255C>T (7566); chr16:31011297; rs2359612; A/G',
    'VKORC1     -1639 consensus',
    'VKORC1 genotype:   497T>G (5808); chr16:31013055; rs2884737; A/C',
    'VKORC1 497 consensus',
    'VKORC1 3730 consensus',
    'VKORC1 1542 consensus',
    'VKORC1     -4451 consensus',
    'VKORC1 genotype:   -4451 C>A (861); Chr16:31018002; rs17880887; A/C',
    'VKORC1 QC genotype:   -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
    # 'Medications', [this ia an repitation of the split dataset]
    # 'Comorbidities',
    'Herbal Medications, Vitamins, Supplements',
    # 'Project Site', coded by PharmGKB
    'Estimated Target INR Range Based on Indication',
    'Carbamazepine (Tegretol)',
    'Rifampin or Rifampicin',
    'Ethnicity (OMB)',
    'Race (OMB)',
    'Acetaminophen or Paracetamol (Tylenol)'
]



# columns_to_select = [
#     'PharmGKB Subject ID',
#     'PharmGKB Sample ID',
#     'Gender',
#     'Age',
#     'Height (cm)',
#     'Weight (kg)',
#     # 'Congestive Heart Failure and/or Cardiomyopathy',
#     # 'Valve Replacement',
#     'Indication for Warfarin Treatment',
#     'Aspirin',
#     'Simvastatin (Zocor)',
#     'Amiodarone (Cordarone)',
#     'Target INR',
#     'Therapeutic Dose of Warfarin',
#     'INR on Reported Therapeutic Dose of Warfarin',
#     'Current Smoker',
#     'Cyp2C9 genotypes',
#     'VKORC1 genotype:   -1639 G>A (3673); chr16:31015190; rs9923231; C/T',
#     'VKORC1 genotype:   3730 G>A (9041); chr16:31009822; rs7294;  A/G',
#     'VKORC1 genotype:   1542G>C (6853); chr16:31012010; rs8050894; C/G',
#     'VKORC1 genotype:   1173 C>T(6484); chr16:31012379; rs9934438; A/G'
# ]


data = data[columns_to_select]


def convert_age_range(age_range):
    if pd.isna(age_range):
        return np.nan
    if age_range == '90+':
        return 90  
    age_parts = age_range.split('-')
    if len(age_parts) == 2:
        lower_bound = int(age_parts[0].strip())
        upper_bound = int(age_parts[1].strip())
        return (lower_bound + upper_bound) / 2
    else:
        return np.nan

def convert_inr_range(inr_range):
    if pd.isna(inr_range):
        return np.nan
    age_parts = inr_range.split('-')
    if len(age_parts) == 2:
        lower_bound = float(age_parts[0].strip())
        upper_bound = float(age_parts[1].strip())
        return (lower_bound + upper_bound) / 2
    else:
        return np.nan


from sklearn.impute import KNNImputer
import pandas as pd

def impute_using_knn(data,coloum, n_neighbors=5):
    # Ensure input is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    y = data[[coloum]]

    # Initialize and apply KNN Imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    y_imputed = imputer.fit_transform(y)

    # Assign imputed values back to the original DataFrame
    data[coloum] = y_imputed

    return data






# Apply the function to the 'Age' column
data['Age'] = data['Age'].apply(convert_age_range)

data['Estimated Target INR Range Based on Indication'] = data['Estimated Target INR Range Based on Indication'].apply(convert_inr_range)





import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import re

def encode_warfarin_indications(indication_column):

    # Replace 'NA' with empty string to handle missing values
    indication_column = indication_column.replace("NA", "")

    # Convert to list of lists (splitting by ';' and 'or')
    indication_list = indication_column.apply(lambda x: re.split(r'\s*;\s*|\s*or\s*|\s*,\s*', str(x)))

    # Multi-label binarizer for one-hot encoding
    mlb = MultiLabelBinarizer()
    encoded_df = pd.DataFrame(mlb.fit_transform(indication_list), columns=mlb.classes_)

    # Rename columns for clarity
    column_mapping = {
        "1": "DVT", "2": "PE", "3": "Afib_flutter", "4": "Heart_Valve",
        "5": "Cardiomyopathy", "6": "Stroke", "7": "Post_Orthopedic", "8": "Other"
    }
    encoded_df = encoded_df.rename(columns=column_mapping)

    return encoded_df


encoded_indications = encode_warfarin_indications(data["Indication for Warfarin Treatment"])
data = pd.concat([data, encoded_indications], axis=1)  # Merge with original DataFrame
data = data.drop(columns=["Indication for Warfarin Treatment"])  # Drop original column if needed


# processing yes/no coloums and replacing NAN with 0.5
medicinal_coloums=['Simvastatin (Zocor)',"Aspirin","Simvastatin (Zocor)","Amiodarone (Cordarone)","Current Smoker","Acetaminophen or Paracetamol (Tylenol)","Rifampin or Rifampicin","Carbamazepine (Tegretol)","Herbal Medications, Vitamins, Supplements"] 

for i in medicinal_coloums:
    data[i] = data[i].replace({'NO': 0, 'YES': 1})
    # data = impute_using_knn(data,i, 5)

    data[i] = data[i].dropna()
    # data[i] = data[i].fillna(0.5)


# Handle missing values (example: fill with mean for numerical columns)

# data = impute_using_knn(data,'Height (cm)', 5)
# data = impute_using_knn(data,'Weight (kg)', 5)


data['Height (cm)'] = data['Height (cm)'].dropna()
data['Weight (kg)'] = data['Weight (kg)'].dropna() # imprives accuracy than imputing



# data['Height (cm)'] = data.groupby('Gender')['Height (cm)'].transform(lambda x: x.fillna(x.mean()))
# data['Weight (kg)'] = data.groupby('Gender')['Weight (kg)'].transform(lambda x: x.fillna(x.mean()))



# data['Gender'] = data['Gender'].dropna()  
data['Gender'] = data['Gender'].fillna('Unknown')  
data['BMI'] = data['Weight (kg)'] / (data['Height (cm)'] / 100) ** 2




# for column in data.columns:
#     unique_values = data[column].unique()
#     print(f"Column: {column}")
#     print(f"Unique values ({len(unique_values)}): {unique_values}\n")



data = pd.get_dummies(data, columns=[
    # 'Indication for Warfarin Treatment',
    'Ethnicity (OMB)','Race (OMB)','VKORC1 QC genotype:   -1639 G>A (3673); chr16:31015190; rs9923231; C/T','VKORC1 genotype:   -4451 C>A (861); Chr16:31018002; rs17880887; A/C','VKORC1     -4451 consensus','VKORC1 1542 consensus','VKORC1 3730 consensus','VKORC1 497 consensus','VKORC1 genotype:   497T>G (5808); chr16:31013055; rs2884737; A/C','VKORC1     -1639 consensus','VKORC1 genotype:   2255C>T (7566); chr16:31011297; rs2359612; A/G','VKORC1 2255 consensus','VKORC1 1173 consensus','Cyp2C9 genotypes','CYP2C9 consensus','Gender', 'VKORC1 genotype:   -1639 G>A (3673); chr16:31015190; rs9923231; C/T','VKORC1 genotype:   3730 G>A (9041); chr16:31009822; rs7294;  A/G','VKORC1 genotype:   1542G>C (6853); chr16:31012010; rs8050894; C/G','VKORC1 genotype:   1173 C>T(6484); chr16:31012379; rs9934438; A/G'], drop_first=True)
data.head()





# Select only numeric columns
numeric_cols = data.select_dtypes(include=['number']).columns

# Compute Q1 (25th percentile) and Q3 (75th percentile) for each numeric column
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1  # Compute IQR for each column

# Identify outliers for each column
outlier_mask = (data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))

# Remove rows with outliers in any numeric column
data_cleaned = data[~outlier_mask.any(axis=1)]

print("Original dataset shape:", data.shape)
print("Cleaned dataset shape:", data_cleaned.shape)

# data=data_cleaned


# numeric_cols = data.select_dtypes(include=[np.number]).columns
# from scipy.stats.mstats import winsorize
# # Apply winsorization to each numeric column (capping extreme 5% on both sides)
# for col in numeric_cols:
#     data[col] = winsorize(data[col], limits=[0.1, 0.1])


from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

# Assuming df is your DataFrame and columns_to_standardize contains the list of numerical columns
columns_to_standardize = ['Age','Height (cm)','Weight (kg)']  # Replace with your column names
scaler = StandardScaler()

# Apply StandardScaler to specified columns
data[columns_to_standardize] = scaler.fit_transform(data[columns_to_standardize])



# data.to_csv("processed.csv",index=False)

for col in data.columns:
    print(f"\nColumn: {col}, Type: {type(data[col])}")
    try:
        print("Unique Values:", data[col].unique())  # Ensure col is a Series
    except Exception as e:
        print(f"\n\nError in column '{col}': {e}\n\n")



# Define features and target
X = data.drop(['Therapeutic Dose of Warfarin'], axis=1)

y = data['Therapeutic Dose of Warfarin'].dropna()

# Align features with non-missing target
X = X.loc[y.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Initialize the XGBoost Regressor with default or tuned hyperparameters
xgb = XGBRegressor(
    learning_rate=0.05,
    max_depth=5,
    n_estimators=100,
    objective='reg:squarederror',
    random_state=42
)





# Convert to NumPy arrays
X_train_array = X_train.to_numpy()
y_train_array = y_train.to_numpy()
X_test_array = X_test.to_numpy()
y_test_array = y_test.to_numpy()



# Fit the model
xgb.fit(X_train_array, y_train_array)

# Make predictions
y_pred = xgb.predict(X_test_array)

# Evaluate the model
mse = mean_squared_error(y_test_array, y_pred)
r2 = r2_score(y_test_array, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)


import matplotlib.pyplot as plt

xgb.fit(X_train, y_train)
plt.barh(X_train.columns, xgb.feature_importances_)
plt.show()

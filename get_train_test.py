import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC
from ED_preprocessing import generate_csv

IN_FILE = 'GeneratedData/fully_processed_ED.csv'
OUT_PATH = 'TrainTestData/'

def main():
    # Load dataset
    df = pd.read_csv(IN_FILE)
    # df = df.head(5000)

    X = df.drop(columns='revisited')
    y = df['revisited']

    # Perform a stratified 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,  # 20% for test set
        stratify=y,     # Maintain the class distribution
        random_state=42 # Seed for reproducibility
    )

    categorical_features = ['gender', 'separation_mode', 'race', 'arrival_mode', 'diagnosis_category', 'age', 'presentation_time', 'ED_LOS',
                        'triage_category', 'triage_pain']
    
    # also add all chief complaint columns to categorical features
    chiefcom_columns = [col for col in X.columns if "chiefcom" in col]
    categorical_features.extend(chiefcom_columns)
   
    numeric_features = [col for col in X.columns if col not in categorical_features]

    # Create imputers
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Fit and transform the train set
    print("Filling in missing values.....\n")
    X_train[numeric_features] = numeric_imputer.fit_transform(X_train[numeric_features])
    X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])

    # Transform the test set using the median from the train set
    X_test[numeric_features] = numeric_imputer.transform(X_test[numeric_features])
    X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])

    X_train['revisited'] = y_train.values
    X_test['revisited'] = y_test.values

    # Print the sizes of the splits
    print(f"Training set size: {X_train.shape}\n")
    print(f"Testing set size: {X_test.shape}")

    generate_csv(X_train, OUT_PATH + 'NO_FS/train.csv')
    generate_csv(X_test, OUT_PATH + 'NO_FS/test.csv')
   

if __name__ == "__main__":
    main()
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
   
    numeric_features = ['n_ed_visits', 'n_ed_admissions', 'triage_temp', 'triage_heartrate', 'triage_resprate', 'triage_o2sat',
				'triage_sbp', 'triage_dbp', 'last_temp', 'last_heartrate', 'last_resprate',
				'last_o2sat', 'last_sbp', 'last_dbp']

    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['number']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

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
    print(f"Training set before SMOTE size: {X_train.shape}\n")

    generate_csv(X_train, OUT_PATH + 'NO_FS/train.csv')
    generate_csv(X_test, OUT_PATH + 'NO_FS/test.csv')

    print("Class distribution in training set before SMOTENC:")
    print(y_train.value_counts())

    y_train = X_train['revisited']
    X_train = X_train.drop(columns='revisited')

    # Get categorical feature indices for SMOTENC
    categorical_features_indices = [list(X_train.columns).index(col) for col in categorical_features]

    # Apply SMOTENC to training set
    print("Applying SMOTENC to training set.....\n")
    smote_nc = SMOTENC(categorical_features=categorical_features_indices, random_state=42)
    X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)

    print("\nClass distribution in trainning set after SMOTENC:")
    print(y_train_resampled.value_counts())

    # merge X and y
    X_train_resampled['revisited'] = y_train_resampled
    generate_csv(X_train_resampled, OUT_PATH + 'NO_FS/train_SMOTE.csv')


    print(f"Training set size after SMOTENC: {X_train_resampled.shape}")
    print(f"Testing set size: {X_test.shape}")

    




if __name__ == "__main__":
    main()
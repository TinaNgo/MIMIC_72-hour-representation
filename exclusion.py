# from https://github.com/nliulab/mimic4ed-benchmark/blob/main/Benchmark_scripts/helpers.py#L315
vitals_valid_range = {
    'temperature': {'outlier_low': 14.2, 'valid_low': 26, 'valid_high': 45, 'outlier_high':47},
    'heartrate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 350, 'outlier_high':390},
    'resprate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 300, 'outlier_high':330},
    'o2sat': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 100, 'outlier_high':150},
    'sbp': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 375, 'outlier_high':375},
    'dbp': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 375, 'outlier_high':375},
}

# Remove entries where the patient is under 18
def remove_under_18(dataframe):
	# Remove patients under 18
	count_under_18 = dataframe[dataframe['age'] < 18].shape[0]
	print(f'Number of entries where age < 18: {count_under_18}')

	print("Removing entries where patient is under 18 years old\n")
	return dataframe[dataframe['age'] >= 18]

# Remove entries where the patient died during the stay or die within 30 days with no re-admission - because they clearly cannot be re-admitted
def remove_died_patients(dataframe):
	count_under_18 = dataframe[dataframe['revisited'] == "DIED"].shape[0]
	print(f'\nNumber of entries where the patient died during the stay or within 30 days of discharge without more re-admission: {count_under_18}')
	print ("\nRemoving entries where the patient died during the stay or within 30 days of discharge without more re-admission...\n")
	return dataframe[dataframe['revisited'] != "DIED"]

# Remove entries with invalid diagnosis category
def remove_invalid_diagnosis(dataframe):
	print("\nRemoving entries with invalid diagnosis category.....\n")
	valid_categories = ['BLD', 'CIR', 'DEN', 'DIG', 'EAR', 'END', 'EXT', 'EYE', 'FAC', 'GEN', 'INF', 'INJ', 'MAL', 'MBD', 'MUS', 'NEO', 'NVS', 'PNL', 'PRG', 'RSP', 'SKN', 'SYM']
	valid_rows = dataframe[dataframe['diagnosis_category'].isin(valid_categories)]

	# Update the original dataframe
	return valid_rows

# Remove patients that have no triage category
def remove_no_triage_category(dataframe):
    print("Removing entries with no triage category.....\n")
    valid_rows = dataframe[dataframe['acuity'].notna()]
    return valid_rows
    
# Set outlier values to null
# Code from https://github.com/nliulab/mimic4ed-benchmark/blob/main/Benchmark_scripts/helpers.py#L315
def outlier_removal_imputation(column_type, vitals_valid_range):
    column_range = vitals_valid_range[column_type]
    def outlier_removal_imputation_single_value(x):
        if x < column_range['outlier_low'] or x > column_range['outlier_high']:
            # set as missing
            return None
        elif x < column_range['valid_low']:
            # impute with nearest valid value
            return column_range['valid_low']
        elif x > column_range['valid_high']:
            # impute with nearest valid value
            return column_range['valid_high']
        else:
            return x
    return outlier_removal_imputation_single_value


# Remove outliers in vital sign data
def remove_outliers(df):
    for column in df.columns:
        column_type = None
        if 'temp' in column:
            column_type = 'temperature'
        elif 'heartrate' in column:
            column_type = 'heartrate'
        elif 'resprate' in column:
            column_type = 'resprate'
        elif 'o2sat' in column:
            column_type = 'o2sat'
        elif 'sbp' in column:
             column_type = 'sbp'    
        elif 'dbp' in column:
             column_type = 'dbp'

        if column_type != None:
            print(f'Removing outliers in \'{column}\' column....\n')
            df[column] = df[column].apply(outlier_removal_imputation(column_type, vitals_valid_range))

    return df

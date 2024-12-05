import pandas as pd
import numpy as np
from pandas import DataFrame
from ED_preprocessing import generate_csv

NON_DISCRETISED_FILE_PATH = "GeneratedData/ED.csv"

numeric_columns = ['n_ed_visits',
				'n_ed_admissions',
				'triage_temp',
				'triage_heartrate',
				'triage_resprate',
				'triage_o2sat',
				'triage_sbp',
				'triage_dbp']

# Discretise patient age into group
# 18-25, 26-45, 46-65, 66-85 and 85+
def discretise_age(dataframe: DataFrame) -> DataFrame:
	print("Discretising age.....\n")
	bins = [17, 25, 45, 65, 85, float('inf')]
	labels = ['18-25', '26-45', '46-65', '66-85', '85+']

	# Discretise the age column
	dataframe['age_group'] = pd.cut(dataframe['age'], bins=bins, labels=labels, right=True)
	return dataframe


# def discretise_temperature(dataframe: DataFrame) -> DataFrame:
# 	print("Discretising temperature.....\n")

# 	# Discretise the patient’s temperature in degrees Farenheit.
# 	bins = [0, 95, 96.8, 100.4, 102.2, float('inf')]
# 	labels = ['≤95', '95.1-96.8', '96.9-100.4', '100.5-102.2', '≥102.2']

# 	dataframe['temperature_group'] = pd.cut(dataframe['temperature'], bins=bins, labels=labels, right=True)
# 	return dataframe


# def discretise_heartrate(dataframe: DataFrame) -> DataFrame:
# 	print("Discretising heart rate.....\n")

# 	bins = [0, 40, 50, 90, 110, 130, float('inf')]
# 	labels = ['≤40', '41-50', '51-90', '91-110', '111-130', '≥131']

# 	dataframe['heartrate_group'] = pd.cut(dataframe['heartrate'], bins=bins, labels=labels, right=True)
# 	return dataframe


# def discretise_resprate(dataframe: DataFrame) -> DataFrame:
# 	print("Discretising respiratory rate.....\n")

# 	bins = [0, 8, 11, 20, 24, float('inf')]
# 	labels = ['≤8', '9-11', '12-20', '21-24', '≥25']

# 	dataframe['resrate_group'] = pd.cut(dataframe['resrate'], bins=bins, labels=labels, right=True)
# 	return dataframe


# def discretise_systolic_bp(dataframe: DataFrame) -> DataFrame:
# 	print("Discretising systolic blood pressure.....\n")

# 	bins = [0, 90, 100, 110, 219, float('inf')]
# 	labels = ['≤90', '91-100', '101-110', '111-219', '≥220']

# 	dataframe['sbp_group'] = pd.cut(dataframe['sbp'], bins=bins, labels=labels, right=True)
# 	return dataframe


# def discretise_o2sat(dataframe: DataFrame) -> DataFrame:
# 	print("Discretising oxygen saturation.....\n")

# 	bins = [0, 91, 93, 95, float('inf')]
# 	labels = ['≤91%', '92-93%', '94-95%', '≥96']

# 	dataframe['o2sat_group'] = pd.cut(dataframe['o2sat'], bins=bins, labels=labels, right=True)
# 	return dataframe

def discretise_presentation_time(df: DataFrame) -> DataFrame:
	print("Discretising presentation time.....\n")
	conditions = [
		(df['presentation_hour'] >= 8) & (df['presentation_hour'] < 17),  # 8am-5pm
		(df['presentation_hour'] >= 17) & (df['presentation_hour'] < 23), # 5pm-11pm
		(df['presentation_hour'] >= 23) | (df['presentation_hour'] < 8)   # 11pm-8am
	]

	labels = ['business hours', 'evening', 'night']
	df['presentation_time'] = np.select(conditions, labels, default='unknown')

	return df


def discretise_LOS(dataframe: DataFrame) -> DataFrame:
	print("Discretising ED Length of stay.....\n")
# Contains categories in 0-4, 5-12, 13-24 and 24+ measured in hours, as per RPA staff’s request.
	bins = [0, 4, 12, 24, float('inf')]
	# labels = ["0-4'", "'5-12'", "'13-24'", "'24+'"]
	labels = ["0-4", "5-12", "13-24", "24+"]

	dataframe['ED_LOS'] = pd.cut(dataframe['LOS (hours)'], bins=bins, labels=labels, right=True)
	return dataframe

def discretise_triage_category(dataframe: DataFrame) -> DataFrame:
	print("Converting triage category to nominal.....\n")

	bins = [0, 1, 2, 3, 4, 5]
	labels = ["one", "two", "three", "four", "five"]

	dataframe['triage_category'] = pd.cut(dataframe['acuity'], bins=bins, labels=labels, right=True)
	return dataframe

def discretise_pain_category(dataframe: DataFrame) -> DataFrame:
	print("Converting triage category to nominal.....\n")

	bins = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

	dataframe['triage_pain_discretised'] = pd.cut(dataframe['triage_pain'], bins=bins, labels=labels, right=True)
	# dataframe['last_pain_discretised'] = pd.cut(dataframe['last_pain'], bins=bins, labels=labels, right=True)
	return dataframe

def normalise(df: DataFrame) -> DataFrame:
	print("Normalising numerical data.....\n")

	
	for col in numeric_columns:
		print("Normalising '" + col + "'column")
		raw_name = 'raw_' + col
		df.rename(columns={col: raw_name}, inplace=True)
		df[col] = (df[raw_name] - df[raw_name].min()) / (df[raw_name].max() - df[raw_name].min())

	return df

def main():
	non_discretised_df = pd.read_csv(NON_DISCRETISED_FILE_PATH)

	discretised_set = discretise_age(non_discretised_df)
	discretised_set = discretise_presentation_time(discretised_set)
	discretised_set = discretise_LOS(discretised_set)
	discretised_set = discretise_triage_category(discretised_set)
	discretised_set = discretise_pain_category(discretised_set)

	discretised_set = normalise(discretised_set)
	# Make the re_admitted column the last column again
	cols = [col for col in discretised_set.columns if col != 'revisited'] + ['revisited']
	discretised_set = discretised_set[cols]

	generate_csv(discretised_set, 'GeneratedData/debug_discritised_ED.csv')

	discretised_set = discretised_set.drop(columns=['age', 'LOS (hours)', 'acuity', 'presentation_hour', 'triage_pain'])

	for col in numeric_columns:
		raw_col = "raw_" + col
		discretised_set = discretised_set.drop(columns=[raw_col])

	discretised_set.rename(columns={'age_group': 'age'}, inplace=True)
	discretised_set.rename(columns={'triage_pain_discretised': 'triage_pain'}, inplace=True)

	generate_csv(discretised_set, 'GeneratedData/fully_processed_ED.csv')
	# unique_values = discretised_set['diagnosis_category'].value_counts().count()

	# print(unique_values)
	# generate_csv(discretised_set.head(50000), 'GeneratedData/crop_discritised_ED.csv')



if __name__ == "__main__":
	main()

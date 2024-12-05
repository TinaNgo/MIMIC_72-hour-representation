import pandas as pd
from ED_preprocessing import generate_csv

DIAGNOSIS_PATH = 'MIMIC-IV_dataset/mimic-iv-ed/ed/diagnosis.csv'
GEMS_PATH = 'diagnosis_gems_2018/2018_I9gem.txt'
CCSR_PATH = 'CCSR/DXCCSR_v2024-1.csv'

# Dictionary to hold ICD-9 to ICD-10 mappings
GEMS_tool = {}
CCSR_tool = {} # Clinical Classifications Software Refined

def load_GEMs():
	global GEMS_tool

	print("Loading GEMs tool.....\n")
	file = open(GEMS_PATH, "r")
    
	for line in file:
		line = line.split()
		
		if line[0] not in GEMS_tool:
			GEMS_tool[line[0]] = line[1]

def load_CCSR():
	global CCSR_tool
	df = pd.read_csv(CCSR_PATH)

	df = df.map(lambda x: x.strip().replace('"', '').replace("'", '') if isinstance(x, str) else x)
	# print(df.head())
	# Create a dictionary with ICD-10-CM CODE as the key and Default CCSR CATEGORY OP as the value
	CCSR_tool = df.set_index('\'ICD-10-CM CODE\'')['\'Default CCSR CATEGORY OP\''].to_dict()


def get_icd10(row):
	if row['icd_version'] == 9:
		return GEMS_tool.get(row['icd_code'], "Error")  # Write 'Error' if such icd code is found
	else:
		return row['icd_code']

def convert_all_to_ICD10(diagnosis):
	print("Converting all diagnosis code to ICD 10.....\n")
	
	diagnosis['icd_10'] = diagnosis.apply(get_icd10, axis=1)
	return diagnosis

def get_ccsr(row):
	return CCSR_tool.get(row['icd_10'], "Error")  # Write 'Error' if such icd code is found

def get_ccsr_category(row):
	return CCSR_tool.get(row['icd_10'], "Error")[:3]

def classify_diagnoses(diagnosis):
	print("Classifying diagnosis using the CCSR.....\n")
	diagnosis['diagnosis_class'] = diagnosis.apply(get_ccsr, axis=1)
	diagnosis['diagnosis_category'] = diagnosis.apply(get_ccsr_category, axis=1)

	return diagnosis
         
# remove rows where icd_10 value is "Error" or "NoDx"
def remove_invalid_rows(diagnosis):
	# Remove rows where icd_10 is "Error" or "NoDx"
	print("Removing invalid rows.....\n")
	filtered_diagnosis_df = diagnosis[~diagnosis['icd_10'].isin(['Error', 'NoDx'])]
	return filtered_diagnosis_df

def main():
	load_GEMs()

	print("Loading diagnosis.csv .....\n")
	diagnosis = pd.read_csv(DIAGNOSIS_PATH)

	# Filter for the highest relevant diagnosis
	print("Filtering for diagnoses with highest relevance.....\n")
	diagnosis = diagnosis[diagnosis['seq_num'] == 1]

	diagnosis = convert_all_to_ICD10(diagnosis)
	diagnosis = remove_invalid_rows(diagnosis)

	# # Count the number of unique values in the 'icd_10' column
	unique_counts = diagnosis['icd_10'].value_counts()
	# print(unique_counts.head(10))
	# # Print the number of unique values
	print(f"Number of unique 'icd_10' codes: {unique_counts.count()}")

	unique_counts = diagnosis['icd_10'].value_counts()

	load_CCSR()
	diagnosis = classify_diagnoses(diagnosis)
	unique_counts = diagnosis['diagnosis_class'].value_counts()
	print(f"Number of unique 'classification' codes: {unique_counts.count()}")

	unique_counts = diagnosis['diagnosis_category'].value_counts()
	print(f"Number of unique 'category' codes: {unique_counts.count()}")

	generate_csv(diagnosis, 'GeneratedData/processed_diagnosis.csv')


if __name__ == "__main__":
    main()
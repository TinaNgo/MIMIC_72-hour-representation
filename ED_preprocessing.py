import pandas as pd
import numpy as np
from exclusion import *
import os
import re

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

EDSTAYS_PATH = 'MIMIC-IV_dataset/mimic-iv-ed/ed/edstays.csv'
PATIENTS_PATH = 'MIMIC-IV_dataset/mimic-iv-hosp-icu/hosp/patients.csv'
TRIAGE_PATH = 'MIMIC-IV_dataset/mimic-iv-ed/ed/triage.csv'
DIAGNOSIS_PATH = 'GeneratedData/processed_diagnosis.csv'
# VITALSIGN_PATH = 'MIMIC-IV_dataset/mimic-iv-ed/ed/vitalsign.csv'
CACHED_N_ED_STAYS = 'GeneratedData/Cached_n_edstays.csv'

def convert_temp_to_celcius(df):
	temp_columns = ['triage_temp']
	
	for col in temp_columns:
		df[col] -= 32
		df[col] *= 5/9
	
	return df

def generate_csv(dataframe, filepath):
	print("Generating csv file: " + filepath + "\n")
	dataframe.to_csv(filepath, index=False)

# Load edstays table from ed modules
def load_edstays():
	print("Loading edstay.csv .....\n")
	edstays = pd.read_csv(EDSTAYS_PATH)

	# drop hadm_id because this is the id for the hospital admission after ed discharged, which we don't need
	edstays = edstays.drop(columns=['hadm_id'])

	# Convert the time to DATETIME object
	edstays['intime'] = pd.to_datetime(edstays['intime'])
	edstays['outtime'] = pd.to_datetime(edstays['outtime'])
	edstays['presentation_hour'] = edstays['intime'].dt.hour

	# sort by subject_id and intime
	edstays = edstays.sort_values(by=['subject_id', 'intime'])

	# calculate the length of stay in hours
	print("Calculating ED length of stay.....\n")
	edstays['LOS (hours)'] = ((edstays['outtime'] - edstays['intime']).dt.total_seconds() / 3600).round()

	# Remove rows where LOS is less than 0
	print("Removing rows where LOS is not positive.....\n")
	edstays = edstays[edstays['LOS (hours)'] > 0]

	return edstays

# Load patients table from hosp module
def load_patients():
	print("Loading patients.csv .....\n")
	patients = pd.read_csv(PATIENTS_PATH)
	patients['dod'] = pd.to_datetime(patients['dod'])

	# remove some columns from the data frame because we don't need them or they are duplicate
	return patients.drop(columns=['gender'])

# Function to check if a pain value is an integer between 0 and 10
def is_valid_pain_value(x):
    try:
        value = int(x)
        return 0 <= value <= 10
    except ValueError:
        return False

# Load triage table from ed module
def load_triage():
	print("Loading triage.csv .....\n")
	triage = pd.read_csv(TRIAGE_PATH)

	triage['temperature'] = pd.to_numeric(triage['temperature'], errors='coerce')
	triage['heartrate'] = pd.to_numeric(triage['heartrate'], errors='coerce')
	triage['resprate'] = pd.to_numeric(triage['resprate'], errors='coerce')
	triage['o2sat'] = pd.to_numeric(triage['o2sat'], errors='coerce')
	triage['sbp'] = pd.to_numeric(triage['sbp'], errors='coerce')
	triage['dbp'] = pd.to_numeric(triage['dbp'], errors='coerce')
	triage['pain'] = pd.to_numeric(triage['pain'], errors='coerce')
	# change all invalid pain value (i.e. not in the range [0,10]) to null
	triage['pain'] = triage['pain'].apply(lambda x: int(x) if is_valid_pain_value(x) else np.nan)
	

	return triage


# Load dianosis table from ed module
def load_diagnosis():
	print("Loading diagnosis.csv .....\n")
	diagnosis = pd.read_csv(DIAGNOSIS_PATH)
	return diagnosis

# Merge edstays data with patients data
def merger_edstays_patients(ed_df, patients_df):
	print("Merging edstays and patients data......\n")
	merged_data = pd.merge(ed_df, patients_df, on='subject_id', how='left')
	merged_data['admission_year'] = pd.to_datetime(merged_data['intime']).dt.year
	merged_data['age'] = merged_data['anchor_age'] + (merged_data['admission_year'] - merged_data['anchor_year'])

	return merged_data.drop(columns=['anchor_age', 'anchor_year', 'anchor_year_group', 'admission_year'])


def merger_edstays_triage(ed_df, triage):
	print("Merging triage data......\n")

	triage_columns = ['subject_id', 'stay_id', 'acuity', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'chiefcomplaint']
	column_rename = {
		'temperature': 'triage_temp',
		'heartrate': 'triage_heartrate',
		'resprate': 'triage_resprate',
		'o2sat': 'triage_o2sat',
		'sbp': 'triage_sbp',
		'dbp': 'triage_dbp',
		'pain': 'triage_pain'
	}

	merged_df = pd.merge(ed_df, triage[triage_columns],  on=['subject_id', 'stay_id'], how='left')
	merged_df.rename(columns=column_rename, inplace=True)
	return merged_df


def merger_edstays_diagnosis(ed_df, diagnosis_df):
	print("Merging edstays and diagnosis data......\n")
	merged_df = pd.merge(ed_df, diagnosis_df[['subject_id', 'stay_id', 'diagnosis_category']],  on=['subject_id', 'stay_id'], how='left')
	return merged_df


def merger_edstays_last_vital(ed_df, last_vitals):
	print("Merging edstays and last recorded vital sign data......\n")

	vital_columns = ['subject_id', 'stay_id', 'last_temp', 'last_heartrate', 'last_resprate', 'last_o2sat', 'last_sbp', 'last_dbp', 'last_pain']

	merged_df = pd.merge(ed_df, last_vitals[vital_columns],  on=['subject_id', 'stay_id'], how='left')
	return merged_df


# Classify each admission as True, False, or DIED, where
# True: re-presented to ED within 72 hours
# False: not re-presented to ED within 72 hours
# DIED: died during stay, or within 72 hours of being discharged from the ED without re-presentation
def classify_admissions(admissions):
    print("\nClassifying admissions as True, False, or DIED...\n")
    # Initialize a list to store the results
    re_admitted_or_died = []
    # for each entry in the admissions table
    for i in range(len(admissions)):
        # if current row is not the last row
        if i < len(admissions) - 1:
            # if the next entry is also for the same patient
            if admissions.loc[i, 'subject_id'] == admissions.loc[i + 1, 'subject_id']:
                # Calculate the time difference in hours
                delta = admissions.loc[i + 1, 'intime'] - admissions.loc[i, 'outtime']
                hours = delta.total_seconds() / 3600
                # if re-admitted within 72 hours
                if 0 <= hours <= 72:
                    re_admitted_or_died.append(True)
                else:
                    re_admitted_or_died.append(False)
            else:  # next entry is not the same patient
                # if day of death is recorded
                if pd.notna(admissions.loc[i, 'dod']):
                    # Calculate the time difference in hours
                    delta = admissions.loc[i, 'dod'] - admissions.loc[i, 'outtime']
                    hours = delta.total_seconds() / 3600
                    # if died within 72 hours of discharge without re-admission
                    if 0 <= hours <= 72:
                        re_admitted_or_died.append("DIED")
                    else:
                        re_admitted_or_died.append(False)
                else:  # no day of death recorded
                    re_admitted_or_died.append(False)
        else:  # last row, only need to check dod
            if pd.notna(admissions.loc[i, 'dod']):
                # Calculate the time difference in hours
                delta = admissions.loc[i, 'dod'] - admissions.loc[i, 'outtime']
                hours = delta.total_seconds() / 3600
                # if died within 72 hours
                if 0 <= hours <= 72:
                    re_admitted_or_died.append("DIED")
                else:
                    re_admitted_or_died.append(False)
            else:  # no day of death recorded
                re_admitted_or_died.append(False)
    admissions['revisited'] = re_admitted_or_died
    return admissions


def process_race(edstays):
	print("Processing patient race.....\n")
	edstays.rename(columns={'race': 'raw_race'}, inplace=True)

	race = []

	for _, stay in edstays.iterrows():
		if "WHITE" in stay['raw_race']:
			race.append("WHITE")
		elif "ASIAN" in stay['raw_race']:
			race.append("ASIAN")
		elif "BLACK" in stay['raw_race']:
			race.append("BLACK")
		elif "HISPANIC" in stay['raw_race']:
			race.append("HISPANIC/LATINO")
		elif stay['raw_race'] == "UNKNOWN" or stay['raw_race'] == "UNABLE TO OBTAIN" or stay['raw_race'] == "PATIENT DECLINED TO ANSWER":
			race.append("UNKNOWN")
		else:
			race.append("OTHER")
				
	edstays['race'] = race
	edstays = edstays.drop(columns=['raw_race'])
	return edstays


def get_separation_mode(edstays):
	print("Getting separation mode.....\n")
	separation_mode = []

	for _, stay in edstays.iterrows():
		if stay['disposition'] == "ADMITTED" or stay['disposition'] == "TRANSFER":
			separation_mode.append("admitted")
		elif stay['disposition'] == "HOME":
			separation_mode.append("discharged")
		elif stay['disposition'] == "LEFT WITHOUT BEING SEEN" or stay['disposition'] == "LEFT AGAINST MEDICAL ADVICE" or stay['disposition'] == "ELOPED":
			separation_mode.append("left without being seen")
		else:
			separation_mode.append("expired/other")
				
	edstays['separation_mode'] = separation_mode
	return edstays

def get_arrival_mode(edstays):
	print("Getting arrival mode.....\n")
	arrival_mode = []

	for _, stay in edstays.iterrows():
		if stay['arrival_transport'] == "AMBULANCE" or stay['arrival_transport'] == "HELICOPTER":
			arrival_mode.append('ambulance')
		elif stay['arrival_transport'] == "WALK IN" or stay['arrival_transport'] == "OTHER":
			arrival_mode.append('other')
		else:
			arrival_mode.append('unknown')
	
	edstays['arrival_mode'] = arrival_mode
	return edstays


def get_n_ed_visits(edstays):
	print("Calculating number of ED visits within the past year.....\n")
	# Initialize the n_edstays column
	edstays['n_ed_visits'] = 0

	# Group by subject_id
	grouped = edstays.groupby('subject_id')

	# Calculate the rolling count of visits within 365 days
	for name, group in grouped:
		group['n_ed_visits'] = group.rolling('365D', on='intime').count()['stay_id'] - 1
		edstays.loc[group.index, 'n_ed_visits'] = group['n_ed_visits']

	return edstays


def get_n_ed_admissions(edstays):
    print("Calculating number of admissions via the ED within the past year.....\n")

    # Initialize the n_ed_admissions column
    edstays['n_ed_admissions'] = 0

    # Group by subject_id
    grouped = edstays.groupby('subject_id')

    for name, group in grouped:
        # Sort the group by 'intime' to ensure correct order
        group = group.sort_values('intime')
        
        # Loop through each stay in the group
        for i in range(len(group)):
            current_intime = group.iloc[i]['intime']
            
            # Define the time window (365 days before the current stay)
            start_window = current_intime - pd.Timedelta(days=365)
            
            # Count the number of admissions in that window
            admissions_count = group[(group['intime'] > start_window) & 
                                     (group['intime'] < current_intime) & 
                                     (group['separation_mode'] == 'admitted')].shape[0]
            
            # Assign the count to the current stay
            edstays.loc[group.index[i], 'n_ed_admissions'] = admissions_count

    return edstays

def get_n_visits_admissions(edstays):
	if os.path.exists(CACHED_N_ED_STAYS):
		print("Getting n_ed_visits and n_ed_stays from memory.....\n")
		n_ed_stays = pd.read_csv(CACHED_N_ED_STAYS)
		merged_df = pd.merge(edstays, n_ed_stays[['subject_id', 'stay_id', 'n_ed_visits', 'n_ed_admissions']],  on=['subject_id', 'stay_id'], how='left')
		return merged_df
		
	edstays = get_n_ed_visits(edstays)
	edstays = get_n_ed_admissions(edstays)
	generate_csv(edstays, CACHED_N_ED_STAYS)

	return edstays


# Code from Xie et al.'s https://github.com/nliulab/mimic4ed-benchmark
def encode_chief_complaint(df_master):
	print("Encoding chief complaints.....\n")

	complaint_dict = {"chiefcom_chest_pain" : "chest pain", "chiefcom_abdominal_pain" : "abdominal pain|abd pain", 
				   "chiefcom_headache" : "headache|lightheaded", "chiefcom_shortness_of_breath" : "breath", "chiefcom_back_pain" : "back pain", "chiefcom_cough" : "cough", 
				   "chiefcom_nausea_vomiting" : "nausea|vomit", "chiefcom_fever_chills" : "fever|chill", "chiefcom_syncope" :"syncope", "chiefcom_dizziness" : "dizz"}
	
	holder_list = []
	complaint_colnames_list = list(complaint_dict.keys())
	complaint_regex_list = list(complaint_dict.values())

	for i, row in df_master.iterrows():
		curr_patient_complaint = str(row['chiefcomplaint'])
		curr_patient_complaint_list = [False for _ in range(len(complaint_regex_list))]
		complaint_idx = 0

		for complaint in complaint_regex_list:
			if re.search(complaint, curr_patient_complaint, re.IGNORECASE):
				curr_patient_complaint_list[complaint_idx] = True
			complaint_idx += 1
		
		holder_list.append(curr_patient_complaint_list)

	df_encoded_complaint = pd.DataFrame(holder_list, columns = complaint_colnames_list)

	df_master = pd.concat([df_master,df_encoded_complaint], axis=1)
	return df_master


def main():
    # Load the edstays table into a DataFrame
	edstays = load_edstays()
	edstays = process_race(edstays)
	edstays = get_separation_mode(edstays)
	edstays = get_arrival_mode(edstays)
	edstays = get_n_visits_admissions(edstays)
	
    # Load and merge patient data
	patients = load_patients()
	edstays = merger_edstays_patients(edstays, patients)

	# Load and merge diagnosis data
	diagnosis = load_diagnosis()
	edstays = merger_edstays_diagnosis(edstays, diagnosis)
	
	# Load and merge triage data
	triage = load_triage()
	edstays = merger_edstays_triage(edstays, triage)
	edstays = encode_chief_complaint(edstays)

	# Convert all temperature values from Fahrenheit to Celcius
	edstays = convert_temp_to_celcius(edstays)

	# Remove entries where the patient is under 18
	edstays = remove_under_18(edstays)

	# Classify each admission as True, False or DIED
	edstays = classify_admissions(edstays)

	# removed entries where the patient died within 72 hours or during the stay
	edstays = remove_died_patients(edstays)

	edstays = remove_invalid_diagnosis(edstays)

	edstays = remove_no_triage_category(edstays)

	edstays = remove_outliers(edstays)

	generate_csv(edstays, 'GeneratedData/ed_admission.csv')

	# Dropping irrelevant columns
	edstays = edstays.drop(columns=['subject_id', 'stay_id', 'intime', 'outtime', 'dod', 'disposition', 'arrival_transport', 'chiefcomplaint'])

	generate_csv(edstays, 'GeneratedData/ED.csv')
	

if __name__ == "__main__":
    main()


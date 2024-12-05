# Predicting 30-day ED representation using Machine Learning
# Dataset: MIMIC-IV

## Process diagnosis.csv file
Run the diagnosis_processing.py script to convert all diagnosis coded to ICD-10 (the raw data include a mixture of both ICD-9 and ICD-10), and classify all diagnoses into categories.

```bash
python3 diagnosis_processing.py
```

## Generate the ED dataset using mainly the files from MIMIC-IV-ED
```bash
python3 ED_preprocessing.py
```

## Discretise and normalise the dataset
```bash
python3 discretise_normalised.py
```

fully_processed_ED.csv is the result dataset.

## Generate full features train and test set
This split fully_processed_ED.csv into train-test set (80-20).

```bash
python3 get_train_test.py
```

## Perform feature selections on the train set

- CFS: "separation_mode", "n_ed_visits"
- Information Gain: "separation_mode", "n_ed_visits", "diagnosis_category", "n_ed_admissions", "triage_category", "revisited"
- Manual Feature Selection: "gender", "separation_mode", "diagnosis_category", "age", "triage_category"


CFS and Information Gain can be performed using Weka, but it can also be done via code. After having the list of selected features, run the code below. \<FS method\>_col_list arrays in this file script are the lists of selected features for each FS method.

```bash
python3 make_FS_set.py
```

## Generate the balanced train sets using SMOTE-NC or SMOTE-N
SMOTE-NC for datasets containing both nominal and continuous features
SMOTE-N for datasets containing only nominal features

```bash
python3 apply_smotenc.py
python3 apply_smoten.py
```


## Convert CSV files to ARFF files
There would be a comparability issue between the train sets and the test sets when evaluating using Weka if we kept the file in CSV format. So we converted all files to ARFF format.

```bash
python3 csv_to_arff.py <config file>
```

The config files can be found in ARFF_converter. 

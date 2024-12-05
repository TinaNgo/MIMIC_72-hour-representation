import pandas as pd
from contextlib import redirect_stdout

TARGET_VAR = 'revisited'

def count_rows_containing_nan(df):
    return df.isnull().any(axis=1).sum()

def output_analytics(df):
    # Count number of instances containing missing values
    if df.isnull().values.any():
        n_missing = count_rows_containing_nan(df)
        n_rows = len(df)
        print(f'{n_missing} rows out of {n_rows} rows '
              f'({n_missing / n_rows * 100:.2f}%) contain missing values.\n')
        df = df.fillna('missing')
    # Cross-tabulate each attribute against class attribute
    for col in df.columns:
        if col == TARGET_VAR:
            continue
        # tab = pd.crosstab(
        #     df[col], df[TARGET_VAR], dropna=False, margins=True)
        tab = pd.crosstab(
            df[col], df[TARGET_VAR], normalize='index', dropna=False, margins=True) * 100  # Show as percentage
        print(tab, '\n')
        
def main():
	df = pd.read_csv("/Volumes/TINA_UNI/DataAndCodeMIMIC/GeneratedData/fully_processed_ED.csv")
	with open("fully_processed_ED " + 'summary1.txt', 'w') as f:
		with redirect_stdout(f):
			output_analytics(df)
        
main()
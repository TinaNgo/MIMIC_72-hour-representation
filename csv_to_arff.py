import csv
import argparse
import os
from configparser import ConfigParser
import helpers

def csv_to_arff(csv_filepath, attribute_filepath, output_arff_filepath, relation_name="dataset"):
    # Read attribute definitions from the attribute file
    with open(attribute_filepath, 'r') as attr_file:
        attribute_definitions = attr_file.read()

    # Open the CSV file and ARFF output file
    with open(csv_filepath, 'r') as csv_file, open(output_arff_filepath, 'w') as arff_file:
        csv_reader = csv.reader(csv_file)

        # Write the relation name and attribute definitions to the ARFF file
        arff_file.write(f"@relation {relation_name}\n\n")
        arff_file.write(attribute_definitions + "\n")

        # Extract the header (column names) from the CSV file
        header = next(csv_reader)

        # Write the data section header in the ARFF file
        arff_file.write("\n@data\n")

        # Write each row of the CSV file to the ARFF file
        for row in csv_reader:
            processed_row = []
            for value in row:
                if ' ' in value:  # If a value contains a space, enclose it in single quotes
                    processed_row.append(f"'{value}'")
                else:
                    processed_row.append(value)
            # Convert each row to a comma-separated string and write to ARFF
            arff_file.write(",".join(processed_row) + "\n")

    print(f"Conversion completed: {output_arff_filepath} created.")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('indir', help="""
		A directory containing a config.ini file. All classifier results will be
		output to this file.
		""")
	args = parser.parse_args()
    
	config_file = helpers.assert_dir_contains_config(args.indir)
    # Load the configuration file
	config = ConfigParser()
	config.optionxform = str  # preserve case in config keys
	config.read(config_file)
      
	# Read the paths from the config file
	csv_filepath = config['meta']['csv_filepath']
	attribute_filepath = config['meta']['attribute_filepath']
	output_arff_filepath = config['meta']['output_arff_filepath']
	relation_name = config.get('meta', 'relation_name', fallback='dataset')

	# Check if paths exist
	if not os.path.exists(csv_filepath):
		raise FileNotFoundError(f"CSV file not found: {csv_filepath}")
	if not os.path.exists(attribute_filepath):
		raise FileNotFoundError(f"Attribute file not found: {attribute_filepath}")

	# Call the csv_to_arff function
	csv_to_arff(csv_filepath, attribute_filepath, output_arff_filepath, relation_name)

if __name__ == "__main__":
    main()

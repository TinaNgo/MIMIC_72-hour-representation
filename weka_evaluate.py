import argparse
import helpers
import os
import sys
import numpy as np

from configparser import ConfigParser
from datetime import datetime
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

N_BOOTSTRAPS = 5000
ALPHA = 0.05  # for 95% confidence

# Function to calculate AUC with bootstrapping for confidence intervals
# code by Jiaying (https://github.sydney.edu.au/jima2199/RPA/blob/main/MIMIC_main.py)
def bootstrap_auc(y_test, y_prob):
    bootstrapped_scores = []
    for i in range(N_BOOTSTRAPS):
        indices = resample(range(len(y_prob)), replace=True)
        if len(np.unique(y_test[indices])) < 2:
            continue
        auc_score = roc_auc_score(y_test[indices], y_prob[indices])
        bootstrapped_scores.append(auc_score)
    sorted_scores = np.sort(bootstrapped_scores)
    lower_bound = np.percentile(sorted_scores, (1 - ALPHA) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + ALPHA) / 2 * 100)
    return np.mean(bootstrapped_scores), lower_bound, upper_bound

def evaluate(train_data, test_data, classifier):
    start_time = datetime.now()
    classifier.build_classifier(train_data)
    end_time = datetime.now()
    build_time = end_time - start_time

    evaluation = Evaluation(train_data)
    start_time = datetime.now()
    evaluation.test_model(classifier, test_data)

    # This next section is Jiaying's code to get auc confidence interval
    y_prob = np.array([classifier.distribution_for_instance(test_data.get_instance(i))[1] for i in range(test_data.num_instances)])
    y_test = np.array([test_data.get_instance(i).get_value(test_data.class_index) for i in range(test_data.num_instances)])
    auc, ci_lower, ci_upper = bootstrap_auc(y_test, y_prob)
    
    end_time = datetime.now()
    evaluation_time = end_time - start_time

    with open(f'{experiment_name}.log', 'w') as f:
        f.write(f"Classifier: {classname}\n")
        f.write(f"Options: {options if options else 'default'}\n")
        f.write(f"Classifier build time (HH:mm:ss): {build_time}\n")
        f.write(f"Evaluation time (HH:mm:ss): {evaluation_time}\n")
        f.write(f"\n")
        f.write(f"{classifier}\n")
        f.write(f"\n")
        f.write(f"{evaluation.summary()}\n")
        f.write(f"AUC: {auc} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})\n")
        f.write(f"{evaluation.class_details()}\n")
        f.write(f"{evaluation.confusion_matrix}\n")
        f.write("\n")
        f.write(evaluation_class_summary(evaluation, 0))
        f.write(evaluation_class_summary(evaluation, 1))


def evaluation_class_summary(evaluation, class_index):
    return (
        f"Class {class_index} details\n"
        f"Area under ROC: {evaluation.area_under_roc(class_index)}\n"
        f"False negatives: {evaluation.num_false_negatives(class_index)}\n"
        f"True negatives: {evaluation.num_true_negatives(class_index)}\n"
        f"False positives: {evaluation.num_false_positives(class_index)}\n"
        f"True positives: {evaluation.num_true_positives(class_index)}\n\n"
    )


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('indir', help="""
    A directory containing a config.ini file. All classifier results will be
    output to this file.
    """)
parser.add_argument('--max-heap-size', default=None)
args = parser.parse_args()
config_file = helpers.assert_dir_contains_config(args.indir)

try:
    config = ConfigParser()
    config.optionxform = str  # preserve case in config keys
    config.read(config_file)
    os.chdir(args.indir)  # treat contents of file relative to config.ini

    train_data_filepath = config['meta']['train_data_path']
    helpers.assert_file_exists(train_data_filepath)

    test_data_filepath = config['meta']['test_data_path']
    helpers.assert_file_exists(test_data_filepath)

    with helpers.JVM(max_heap_size=args.max_heap_size):
        # train_data = helpers.load_csv(train_data_filepath)
        # test_data = helpers.load_csv(test_data_filepath)

        train_data = helpers.load_arff(train_data_filepath)
        test_data = helpers.load_arff(test_data_filepath)        
        

        for section in config:
            experiment_name = section

            # skip experiment if it has previously run
            # if os.path.exists(f'{experiment_name}.log'):
            #     continue
            
            experiment = config[section]
            classname = experiment.get('classname', None)
            if classname is None:
                continue
            options = experiment.get('options', "")
            split_string = r' \\ ' if r' \\ ' in options else ' '
            classifier = Classifier(
                classname=classname,
                options=options.split(split_string))

            evaluate(train_data, test_data, classifier)

except Exception as e:
    with open('error.log', 'w') as f:
        f.write(str(e))
    sys.exit(1)
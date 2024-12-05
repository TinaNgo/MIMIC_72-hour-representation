import pandas as pd

edstays = pd.read_csv("GeneratedData/fully_processed_ED.csv")

edstays_no_na = edstays.dropna()
edstays_no_triage_pain = edstays.drop(columns=['triage_pain']).dropna()
print("Original number of samples: ", len(edstays))
print("Num of samples after removing all na:", len(edstays_no_na))
percent = (1 - (len(edstays_no_na)/len(edstays)))*100
print("percentage:", percent)
# print("Num of samples after dropping last_temp column then removing all na:", len(edstays_no_triage_pain))

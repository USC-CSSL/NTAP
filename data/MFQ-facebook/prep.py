import pandas as pd
import sys 
import re
import json
import os
from unidecode import unidecode
from nltk.corpus import wordnet
import nltk
from textblob import TextBlob

source = sys.argv[1]
dest = sys.argv[2]
param_file = sys.argv[3]

with open(param_file, 'r') as fo:
    params = json.load(fo)

raw_df = pd.read_csv(source, delimiter='|', encoding="latin-1", low_memory=False)
print("Extracting dataframe from {0:s} with {1} rows and {2} columns".format(source, raw_df.shape[0], raw_df.shape[1]))

raw_df.fillna(value=-1, inplace=True)
# mfq_avgs = [col for col in raw_df.columns.values if col.startswith("MFQ") and col.endswith("AVG")]
full_data = raw_df.loc[(raw_df["fb_status_msg"] != -1)]

print("Adding contraints: all rows must have text")
print("Removed {} rows".format(len(raw_df) - len(full_data)))
geo_cols = ["country_current", "county_code", "hometown", "zipcode"]
demographics = ["politics" + str(int_val) + "way" for int_val in [10,7,4] ] + \
        ["age", "concentrations", "degrees", "significant_other_id", "relation_status",
         "religion_now", "politics_new", "sex", "gender"]
demo_race = ["race_" + col for col in ["black", "eastasian", "latino", "middleeast", "nativeamerican", "southasian", "white", "other", "decline"]]

demographics = geo_cols + demographics + demo_race
outcomes = [col for col in full_data.columns.values if (col.startswith("MFQ") and col not in ["MFQ_HFminusIAP", "MFQ_BLANKS_perc", "MFQ_failed"]) or col == "mfq_unclear"]

full_selected = full_data[["userid", "fb_status_msg"] + demographics + outcomes]
print("Discarding columns with no useful information.\nUpdated dataframe has {} columns".format(full_selected.shape[1]))
dup_users = [1420984, 1452611, 1420855, 1417857, 1383870, 1425518, 1433151, 1442980, 1428919, 1444426, 1429081, 1392082, 1419999, 1436318, 1429429, 1430001, 1434875, 1442354, 1423338, 1439563, 1418381, 1458417, 1435424, 1456133, 1442353, 1420592, 1411379, 1423476, 1451372, 1402802, 1444236, 1398441, 1443010, 1413376, 1410964, 1432369, 1440158, 1420837, 1425041, 1431418, 1432509, 1444518, 1441000, 1451830, 1432079, 1442376, 1437827, 1447248, 1437095, 1416194, 1405356, 1444711, 1426884, 1421650, 1438806, 1437287, 1456437, 1427897, 1425954, 1412338, 1451418, 1415309, 1442972, 1436154, 1442492, 1421404, 1415559, 1455841, 1456495, 1418103, 1440553, 1455399, 1425386, 1420768, 1433676, 1453411, 1428886, 1442161, 1436849, 1433716, 1439268, 1425522, 1426627, 1440262, 1457500, 1420444, 1417416, 1412465, 1426342, 1435059, 1457980, 1424649, 1440900, 1430534, 1450678, 1439315, 1411251, 1431943, 1416540, 1427043, 1436933, 1458509, 1391258, 1415365, 1431417, 1419078, 1438446, 1442545, 1422034, 1417770, 1422056, 1447222, 1405756, 1424874, 1435902, 1447340, 1448140, 1433004, 1414929, 1446860, 1445592, 1418344, 1436150, 1406328, 1424066, 1458461, 1300486, 1427860, 1422872, 1434405, 1416603, 1414139, 1417440, 1442154, 1423841, 1407551, 1439273, 1449171, 1412897, 1443517, 1426157, 1395378, 1449476, 1425171, 1435689, 1452177, 1451625, 1456357, 1420189, 1436297, 1412493, 1424378, 1426267, 1443957, 1413056, 1417631, 1430291, 1435291, 1417990, 1422552, 1415218, 1425547, 1433584, 1430679, 1431526, 1458300, 1447637, 1428684, 1446629, 1398189, 1420930, 1428932, 1434239, 1417697, 1395213, 1407341, 1416966, 1419941, 1445028]
dropped_df = full_selected[~full_selected["userid"].isin(dup_users)]
dropped_df.to_pickle(dest)

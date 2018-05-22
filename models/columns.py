
geo_cols = ["country_current", "county_code", "hometown", "zipcode"]
demographics = ["politics" + str(int_val) + "way" for int_val in [10,7,4] ] + \
        ["age", "concentrations", "degrees", "significant_other_id", "relation_status", \
         "religion_now", "politics_new", "sex", "gender"]
demo_race = ["race_" + col for col in ["black", "eastasian", "latino", "middleeast", "nativeamerican", "southasian", "white", "other", "decline"] ]

demographics = ["age", "gender"]  # geo_cols + demographics + demo_race

MFQ_AVG = ['MFQ_INGROUP_AVG', 'MFQ_AUTHORITY_AVG', 'MFQ_HARM_AVG', 'MFQ_PURITY_AVG', 'MFQ_FAIRNESS']

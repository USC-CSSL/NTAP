
# make concatenated_dataframe
print("Making dataframe with one row per user (concatenate statuses)")
users = dict()
for index, row in full_selected.iterrows():
    userid = row["userid"]
    if userid not in users:
        users[userid] = [row.to_dict()]
    else:
        users[userid].append(row.to_dict())

new_data = list()
for userid in users:
    texts = list()
    for row in users[userid]:
        texts.append(row["fb_status_msg"])
    users[userid][0]["fb_status_msg"] = texts
    new_data.append(users[userid][0])

df = pd.DataFrame(new_data)







def split_indiv(df, col):
    new_rows = list()
    for index, row in df.iterrows():
        texts = row[col]
        template = row.to_dict()
        for text in texts:
            temp = {k:v for k,v in template.items()}
            temp[col] = text
            new_rows.append(temp)
    return pd.DataFrame(new_rows)

def split_bags(df, col, size_):
    new_rows = list()
    for index, row in df.iterrows():
        texts = row[col]
        template = row.to_dict()
        idx = len(texts)
        while idx >=0:
            new_doc = " ".join(texts[idx - size_: idx])
            idx = max(0, idx - size_)
            template[col] = new_doc
            new_rows.append(template)
    return pd.DataFrame(new_rows)

def restructure(df, col, config_text, data_dir, size_=10):
    if config_text == 'indiv':
        p = os.path.join(data_dir, config_text + ".pkl")
        if os.path.isfile(p):
            return pd.read_pickle(p)
        else:
            df = split_indiv(df, col)
            df.to_pickle(p)
            return df
    elif config_text == 'concat':
        print("Not done")
        exit(1)
    elif config_text == 'rebagged':
        return split_bags(df, col, size_)
    else:
        print("config_text has improper value; exiting")
        return

import emoji, json, os, re, requests
import pandas as pd

emojis = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
emoji_re = re.compile('|'.join(re.escape(p) for p in emojis))

"""def pos(self):
    processed, tokens = list(), list()
    self.corenlp_props['annotators'] = 'pos'
    pbar = ProgressBar()
    for doc in pbar(self.data[self.text_col].values):
        annotated = json.loads(self.corenlp.annotate(doc,
                                    properties=self.corenlp_props))
        POSs, words = list(), list()
        for sent in annotated["sentences"]:
            for token in sent["tokens"]:
                POSs.append(token["pos"])
                words.append(token["word"])
        processed.append(" ".join(POSs))
        tokens.append(" ".join(words))
    self.data.loc[:, 'tokens'] = pd.Series(tokens, index=self.data.index)
    self.data.loc[:, 'pos'] = pd.Series(processed, index=self.data.index)

def ner(self):
    processed = list()
    self.corenlp_props['annotators'] = 'ner'
    pbar = ProgressBar()
    for doc in pbar(self.data[self.text_col].values):
        annotated = json.loads(self.corenlp.annotate(doc,
                            properties=self.corenlp_props))
        doc_ner = list()
        for sent in annotated["sentences"]:
            for item in sent["tokens"]:
                word, ner = item['word'], item["ner"]
                doc_ner.append( (word, ner) )
        processed.append(doc_ner)
    self.data.loc[:, 'ner'] = pd.Series(processed, index=self.data.index)

def depparse(self):
    print("Not Implemented (dependency parsing)")

    #tagme version 1
    def tagme_v1(self, p=0.1):
        #NOTES:
        #       - If parallelizing, limit to 4 concurrent calls
        #       - Pause (10s) periodically on each thread
        token(str): tagme GCUBE token (requires registration)
        entity_dir = os.path.join(self.dest, "tagme")
        if not os.path.isdir(entity_dir):
            os.makedirs(entity_dir)
        entities = dict()
        extracted = list()
        params = {'gcube-token': self.tagme_token,
                  'lang': 'en',
                  'include_abstract': 'true',
                  'include_categories': 'true'
                 }
        abstract_file = open(os.path.join(entity_dir, "abstracts.tsv"), 'a+')
        abstract_file.seek(0)
        category_file = open(os.path.join(entity_dir, "categories.tsv"), 'a+')
        for line in abstract_file:
            if len(line.strip()) > 0:
                id_, title, _ = line.split('\t')
                entities[id_] = title

        request_str = "https://tagme.d4science.org/tagme/tag"
        count = 0
        try:
            pbar = ProgressBar()
            for doc in pbar(self.data[self.text_col].values):
                count += 1
                if count % 200 == 0:
                    # sleep(5)  # sleep for 5 seconds
                    print("Processed {} docs".format(count))
                    abstract_file.flush()
                    category_file.flush()
                row = list()
                params["text"] = doc
                res = requests.get(request_str, params=params)
                if res.status_code == 200:
                    try:
                        ann = res.json()["annotations"]
                        filt = [entry for entry in ann if entry["link_probability"] > p]
                        for filtered in filt:
                            if filtered["id"] not in entities:
                                entities[filtered["id"]] = filtered["title"]
                                # write entity to file with abstract
                                abstract_file.write("{}\t{}\t{}".format(
                                        filtered["id"], filtered["title"],
                                        filtered["abstract"].replace('\t', '')))
                                abstract_file.write('\n')
                                for cat in filtered["dbpedia_categories"]:
                                    category_file.write("{}\t{}".format(filtered["id"], cat))
                                    category_file.write("\n")
                            # write entity info to list, to save in self.data
                            row.append( (filtered["id"], filtered["start"], filtered["end"], filtered["link_probability"]) )
                    except Exception as e:
                        pass
                else:
                    row.append(("No Response",) )
                extracted.append(row)
        except KeyboardInterrupt:
            partial_index = list(self.data.index)[:len(extracted)]
            saved_data = pd.Series(extracted, index=partial_index)
            print("Interrupted; saved to file")
            saved_data.to_pickle(os.path.join(entity_dir, "saved.pkl"))
            return
        self.data.loc[:, "tagme_entities"] = pd.Series(extracted, index=self.data.index)

    def concat_unique_entity(self, entities, abstract):
        agg_entity = []
        if entities == '':
            return ''
        for entity in entities:
            id = entity[1]
            record = abstract.loc[abstract['Id'] == id]
            title = record['Abstract'].iloc[0]
            if title not in agg_entity:
                agg_entity.append(title)
        agg_str = ' '.join(agg_entity)
        return agg_str


    def aggregate_entities(self, abstract_file_path):
        source = pd.read_csv(abstract_file_path, delimiter='\t', quoting=3, names=['Id', 'Title', 'Abstract'])
        self.data['tagme_aggregated_abstract'] = self.data['tagme_entities'].apply(self.concat_unique_entity, abstract=source)

    def tagme_helper(self, doc, request_str, params, abstract_file, category_file, entities, extracted, p):
        row = list()
        params["text"] = doc[1]
        index = doc[0]
        res = requests.get(request_str, params=params)
        if res.status_code == 200:
            try:
                ann = res.json()["annotations"]
                filt = [entry for entry in ann if entry["link_probability"] > p]
                for filtered in filt:
                    if filtered["id"] not in entities:
                        entities[filtered["id"]] = filtered["title"]
                        # write entity to file with abstract
                        abstract_file.write("{}\t{}\t{}".format(
                            filtered["id"], filtered["title"],
                            filtered["abstract"].replace('\t', '')))
                        abstract_file.write('\n')
                        for cat in filtered["dbpedia_categories"]:
                            category_file.write("{}\t{}".format(filtered["id"], cat))
                            category_file.write("\n")
                    # write entity info to list, to save in self.data
                    row.append((index, filtered["id"], filtered["start"], filtered["end"], filtered["link_probability"]))
            except Exception as e:
                pass
        else:
            row.append(("No Response",))
        if row:
            extracted.append(row)
        return extracted

    def threader(self, request_str, params, abstract_file, category_file, entities, extracted, p, entity_dir):
        while True:
            # gets a worker from the queue
            worker = queue.get()

            # Run the example job with the avail worker in queue (thread)
            self.tagme_helper(worker, request_str, params, abstract_file, category_file, entities, extracted, p)

            #print thread name and doc
            with print_lock:
                print(threading.current_thread().name, worker)

            # completed with the job
            queue.task_done()

    # tagme version 2
    def tagme(self, p=0.1):
        entity_dir = os.path.join(self.dest, "tagme")
        if not os.path.isdir(entity_dir):
            os.makedirs(entity_dir)
        entities = dict()
        extracted = list()
        params = {'gcube-token': self.tagme_token,
                  'lang': 'en',
                  'include_abstract': 'true',
                  'include_categories': 'true'
                 }
        abstract_file = open(os.path.join(entity_dir, "abstracts.tsv"), 'a+')
        abstract_file.seek(0)
        category_file = open(os.path.join(entity_dir, "categories.tsv"), 'a+')
        for line in abstract_file:
            if len(line.strip()) > 0:
                id_, title, _ = line.split('\t')
                entities[id_] = title

        request_str = "https://tagme.d4science.org/tagme/tag"

        start = time.time()
        index = list(range(0, len(self.data[self.text_col])))
        index_dict = dict(zip(index, self.data[self.text_col].values))
        for key, value in index_dict.items():
            queue.put((key, value))

        #intializing 4 threads
        try:
            num_of_threads = 4
            for x in range(num_of_threads):
                t = threading.Thread(target=self.threader, args = (request_str, params, abstract_file, category_file, entities, extracted, p, entity_dir))
                t.daemon = True
                t.start()
            queue.join()

        except KeyboardInterrupt:
            partial_index = list(self.data.index)[:len(extracted)]
            saved_data = pd.Series(extracted, index=partial_index)
            print("Interrupted; saved to file")
            saved_data.to_pickle(os.path.join(entity_dir, "saved.pkl"))
            return
        self.data.loc[:, "tagme_entities"] = pd.Series(extracted, index=[i[0][0] for i in extracted])
        self.data['tagme_entities'].fillna('', inplace=True)
        abstract_file.close()
        category_file.close()
        self.aggregate_entities(abstract_file.name)
        print('Time taken by tagme job(sec):', time.time() - start)



# Incorporate into class:

###############################################################################
"""
def emojis(df, col):
    emojis_col = list()
    for i, row in df.iterrows():
        text = row[col]

        this_emojis = r.findall(text)
        text = re.sub(emoji_pattern, "", text)

        for emo in emot.emoticons(text):
            if len(emo['value']) > 1:
                this_emojis.append(emo['value'])
                text = text.replace(emo['value'], "")
        emojis_col.append(this_emojis)
        df.at[i, col] = text
    df["emojis"] = pd.Series(emojis_col, index=df.index)
    return df
"""

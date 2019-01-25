import json
from collections import defaultdict


def el_append_properties(el_file, dbpedia_file, output_file):
    """Append entity properties in DBpedia.
    Keep the following relations:
    - purpose
    - office
    - background
    - meaning
    - orderInOffice
    - seniority
    - title
    - role
    
    :param el_file: Path to the TagMe result file. 
    :param dbpedia_file: Path to DBpedia's mappingbased_literals_en.ttl file.
    :param output_file: Path to the output file.
    """
    import re
    titles = set()
    relations = {'purpose', 'office', 'background', 'meaning', 'orderInOffice',
                 'seniority', 'title', 'role'}
    with open(el_file, 'r', encoding='utf-8') as r:
        for line in r:
            tid, result = line.rstrip().split('\t')
            result = json.loads(result)
            for anno in result['annotations']:
                if 'title' in anno:
                    titles.add(anno['title'])
    print('#title:', len(titles))

    title_properties = defaultdict(list)
    with open(dbpedia_file, 'r', encoding='utf-8') as r:
        for line in r:
            if line.startswith('#'):
                continue
            segs = line[:-2].split(' ', 2)
            if not segs[1].startswith('<http://dbpedia.org/ontology/'):
                continue
            title = segs[0][29:-1].replace('_', ' ')
            relation = segs[1][29:-1]
            if title in titles and relation in relations:
                property = segs[2][segs[2].find('"') + 1:segs[2].rfind('"')]
                property = re.sub(r'(\d+th)', r'\1 ', property)
                property = re.sub(r'(\d+nd)', r'\1 ', property)
                property = re.sub(r'(\d+st)', r'\1 ', property)
                property = re.sub(r'([a-z])([A-Z])', r'\1 \2', property)
                property = re.sub(r'\s+', r' ', property)
                title_properties[title].append(property)

    with open(el_file, 'r', encoding='utf-8') as r, \
            open(output_file, 'w', encoding='utf-8') as w:
        for line in r:
            tid, result = line.rstrip().split('\t')
            result = json.loads(result)
            annotations = []
            for anno in result['annotations']:
                if 'title' in anno:
                    title = anno['title']
                    if title in title_properties:
                        props = '. '.join(title_properties[title])
                        anno['abstract'] += ' ' + props
                    annotations.append(anno)
            result['annotations'] = annotations
            w.write('{}\t{}\n'.format(tid, json.dumps(result)))


def el_clean_by_type(el_file, output_file, dbpedia_file, type_file):
    """Clean the TagMe result by type.
    
    :param el_file: Path to the TagMe result file.
    :param output_file: Path to the output file.
    :param dbpedia_file: Path to DBpedia's instance_types_en.ttl file.
    :param type_file: Path to a file containing types to keep. Each line a type.
    """
    # Load involved entities
    seen_titles = set()
    with open(el_file, 'r', encoding='utf-8') as r:
        for line in r:
            tid, result = line.rstrip().split('\t')
            result = json.loads(result)
            annotations = result['annotations']
            for anno in annotations:
                if 'title' in anno:
                    title = anno['title'].replace(',', '')
                    seen_titles.add(title)

    # Load categories
    title_type = {}
    with open(dbpedia_file, 'r', encoding='utf-8') as r:
        for line in r:
            if line.startswith('#'):
                continue
            segs = line.rstrip().split(' ')
            if len(segs) != 4:
                continue
            title = segs[0][1:-1]
            if not title.startswith('http://dbpedia.org/resource/'):
                continue
            title = title[28:].replace('_', ' ')
            if title in seen_titles:
                entity_type = segs[2]
                if not entity_type.startswith('<http://dbpedia.org/ontology/'):
                    continue
                title_type[title] = entity_type[29:-1]
    print(len(seen_titles))
    print(len(title_type))

    # Load removed categories
    kept = set()
    with open(type_file, 'r', encoding='utf-8') as r:
        for line in r:
            kept.add(line.split('\t')[0])
    print(kept)

    with open(el_file, 'r', encoding='utf-8') as r, \
        open(output_file, 'w', encoding='utf-8') as w:
        for line in r:
            tid, result = line.rstrip().split('\t')
            result = json.loads(result)
            annotations = result['annotations']
            filtered_annotations = []
            for anno in annotations:
                if 'title' in anno:
                    title = anno['title'].replace(',', '')
                    if title in title_type:
                        category = title_type[title]
                        if category in kept:
                            filtered_annotations.append(anno)
                        else:
                            print('removed', category)
                    else:
                        pass
            result['annotations'] = filtered_annotations
            w.write('{}\t{}\n'.format(tid, json.dumps(result)))


def el_clean_by_pos(tweet_file, el_file, output_file, skip_first=False):
    """Clean the entity linking result by POS.
    Keep: NN, NNS, NNPS, NNP, JJ
    
    :param tweet_file: Path to the TSV format tweet data set. 
    :param el_file: Path to the TagMe result file.
    :param output_file: Path to the output file.
    :param skip_first: Skip the first line of <tweet_file> (default=false).
    """
    from nltk import pos_tag, TweetTokenizer
    import nltk
    nltk.download('averaged_perceptron_tagger')

    tokenizer = TweetTokenizer()
    keep_set = {'NN', 'NNS', 'NNPS', 'NNP', 'JJ'}

    # Load tweets
    tid_tokens = {}
    tag_set = set()
    with open(tweet_file, 'r', encoding='utf-8') as r:
        if skip_first:
            next(r)
        for line in r:
            tid, tweet, _ = line.rstrip().split('\t')
            tokens = tokenizer.tokenize(tweet)
            tokens = pos_tag(tokens)
            tid_tokens[tid] = tokens
            for t, p in tokens:
                tag_set.add(p)

    with open(el_file, 'r', encoding='utf-8') as r, \
        open(output_file, 'w', encoding='utf-8') as w:

        for line in r:
            tid, result = line.rstrip().split('\t')
            result = json.loads(result)
            tokens = tid_tokens[tid]
            keep_tokens = set()
            for token, pos in tokens:
                if pos in keep_set:
                    keep_tokens.add(token.lower().replace('#', ''))
            filtered_annotations = []
            if 'annotations' in result:
                for anno in result['annotations']:
                    if 'title' in anno and 'spot' in anno:
                        spot = anno['spot']
                        if len(spot) == 1:
                            continue
                        if ' ' in spot or spot.lower() in keep_tokens:
                            filtered_annotations.append(anno)
            result['annotations'] = filtered_annotations
            w.write('{}\t{}\n'.format(tid, json.dumps(result)))
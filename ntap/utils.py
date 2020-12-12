import zipfile

"""
if not os.path.exists(dictionary_path):
    raise ValueError("Dictionary not found at {}".format(dictionary_path))
if dictionary_path.endswith(".json"):
    try:
        with open(dictionary_path, 'r') as fo:
            dictionary = json.load(fo)  # {category: words}
            categories, items = zip(*sorted(dictionary.items(), key=lambda x:x[0]))
            return categories, items
    except Exception:
        raise ValueError("Could not import json dictionary")
"""

def load_glove(p):
    pass

def load_fasttext(p):
    pass

def load_imdb(file_name):
    d = {'pos': list(), 'neg': list()}
    zf = zipfile.ZipFile(file_name, 'r')
    for name in zf.namelist():
        if name.startswith('pos') and name.endswith('txt'):
            data = zf.read(name)
            d['pos'].append(data.strip())
        if name.startswith('neg') and name.endswith('txt'):
            data = zf.read(name)
            d['neg'].append(data.strip())
    return d



def read_dictionary(liwc_file):
    cates = {}
    words = {}
    percent_count = 0

    for line in liwc_file:
        line_stp = line.strip()
        if line_stp:
            parts = line_stp.split('\t')
            if parts[0] == '%':
                percent_count += 1
            else:
                if percent_count == 1:
                    cates[parts[0]] = parts[1]
                    words[parts[0]] = []
                else:
                    for cat_id in parts[1:]:
                        words[cat_id].append(parts[0])
    items = []
    categories = []
    for cat_id in cates:
        categories.append(cates[cat_id])
        items.append(words[cat_id])
    return tuple(categories), tuple(items)


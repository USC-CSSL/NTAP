import zipfile
import pandas as pd

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

import urllib.request as urllib
from urllib.request import urlopen

"""
class NTAPDownloader:

    links = {
        'glove-wiki': "http://nlp.stanford.edu/data/glove.6B.zip",
         'fasttext-wiki': ("https://dl.fbaipublicfiles.com/"
                           "fasttext/vectors-english/"
                           "wiki-news-300d-1M.vec.zip"),
         'fasttext-wiki-subword': ("https://dl.fbaipublicfiles.com/"
                                   "fasttext/vectors-english/"
                                   "wiki-news-300d-1M-subword.vec.zip"),
         'fasttext-cc': ("https://dl.fbaipublicfiles.com/"
                         "fasttext/vectors-english/"
                         "crawl-300d-2M.vec.zip"),
         'fasttext-cc-subword': ("https://dl.fbaipublicfiles.com/"
                                 "fasttext/vectors-english/"
                                 "crawl-300d-2M-subword.zip") 
    }

    def __init__(self):

        self.base_dir = os.path.expanduser("~/ntap_data")
        self.base_dir = os.environ.get('NTAP_DATA_DIR', self.base_dir)

        #_PARENT_DIR = 

        if not os.path.isdir(BASE_DIR):
            try:
                logger.info("Creating %s", BASE_DIR)
                os.makedirs(BASE_DIR)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    raise Exception(
                        "Not able to create folder ntap_data in {}. File ntap_data "
                        "exists in the directory already.".format(_PARENT_DIR)
                    )
                else:
                    raise Exception(
                        "Can't create {}. Make sure you have the read/write permissions "
                        "to the directory or you can try creating the folder manually"
                        .format(self.base_dir)
                    )


    def download(self, name_of_embedding):

        if name_of_embedding not in self.links:
            raise ValueError(f'{name_of_embedding} not available')

"""

def open_embed(embed_name):
    """

    Check if exists in local project directory
    Check is embed file exists (in ntap data directory).
    If not, initiate download
    """

    if embed_name == 'glove-wiki':
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


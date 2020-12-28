import os
import shutil
import tempfile
import io
import datetime
import zipfile
import pandas as pd
import logging
import requests
from tqdm import tqdm
import numpy as np

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

def load_fasttext_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = np.zeros((n, d))
    vocab = [""] * n
    for i, line in tqdm(enumerate(fin), total=n, desc="load_fasttext"):
        tokens = line.rstrip().split(' ')
        vocab[i] = tokens[0]
        data[i, :] = np.array([float(t) for t in tokens[1:]])
    return vocab, data

def load_glove_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


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

    file_names = {
        'glove-wiki': "glove.6B.{}d.txt",
        'fasttext-wiki': "wiki-news-300d-1M.vec",
        'fasttext-wiki-subword': "wiki-news-300d-1M-subword.vec",
        'fasttext-cc': "crawl-300d-2M.vec",
        'fasttext-cc-subword': "crawl-300d-2M-subword.vec"
    }

    converters = {
        'fasttext': load_fasttext_vectors,
        'glove': load_glove_vectors
    }

    def __init__(self):

        self.base_dir = os.path.expanduser("~/ntap_data")
        self.base_dir = os.environ.get('NTAP_DATA_DIR', self.base_dir)
        self.log = logging.getLogger('downloader')

        if not os.path.isdir(self.base_dir):
            try:
                self.log.info(f'Creating {self.base_dir}')
                os.makedirs(self.base_dir)
            except OSError as e:
                self.log.exception("Can't create {}. ".format(self.base_dir))

    def extract(self, name, path_to_embedding_zip, vec_size=None):
        # verify zip file is there

        if not os.path.exists(path_to_embedding_zip):
            self.log.error(f"Could not find {path_to_embedding_zip}")
            return

        vec_file_name = self.file_names[name]

        if vec_size is not None and "{}" in vec_file_name:
            vec_file_name = vec_file_name.format(vec_size)

        with zipfile.ZipFile(path_to_embedding_zip, 'r') as zip_ref:
            try:
                vec_meta = zip_ref.getinfo(vec_file_name)
            except KeyError:
                self.log.exception(f"Could not find {vec_file} "
                                    "in {path_to_embedding_zip}")
                return
            else:
                with tempfile.TemporaryDirectory() as dirpath:
                    tmp_vec_path = zip_ref.extract(vec_file_name, path=dirpath)
                    converter_name = 'fasttext' if 'fasttext' in name else 'glove'
                    converter = self.converters[converter_name]
                    vocab, data = converter(tmp_vec_path)

                local_vocab_file = os.path.join(self.base_dir, name, "vocab.txt")
                with open(local_vocab_file, 'w') as fo:
                    fo.write('\n'.join(str(w) for w in vocab))
                np.save(os.path.join(self.base_dir, name, 'vecs.npy'), data)
        #mod_date = datetime.datetime(*vec_meta.date_time)

    def is_converted(self, name):

        local_vocab_file = os.path.join(self.base_dir, name, "vocab.txt")
        return os.path.exists(local_vocab_file)

    def load(self, name):

        local_vocab_file = os.path.join(self.base_dir, name, "vocab.txt")
        local_vecs_file = os.path.join(self.base_dir, name, "vecs.npy")
        with open(local_vocab_file, 'r') as fo:
            vocab = [line.strip() for line in fo if line.strip() != ""]
        vecs = np.load(local_vecs_file)

        return vocab, vecs

    def download(self, name_of_embedding):

        if name_of_embedding not in self.links:
            raise ValueError(f'{name_of_embedding} not available')

        local_dest_dir = os.path.join(self.base_dir, name_of_embedding)
        if not os.path.isdir(local_dest_dir):
            self.log.info(f'Creating {local_dest_dir}')
            os.makedirs(local_dest_dir)

        download_loc = self.links[name_of_embedding]
        base_name = os.path.basename(download_loc)
        local_download_loc = os.path.join(local_dest_dir, base_name)

        if os.path.exists(local_download_loc):
            self.log.info("File found on local system; skipping download") 
            return local_download_loc

        self.log.info(f'Downloading {name_of_embedding}')
        url = self.links[name_of_embedding]
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size= 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_download_loc, 'wb') as fo:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                fo.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            self.log.error("Could not download {name_of_embedding}"
                           "from {download_loc}")
        return local_download_loc

def open_embed(embed_name, vec_size=None):
    """

    Check if exists in local project directory
    Check is embed file exists (in ntap data directory).
    If not, initiate download
    """

    downloader = NTAPDownloader()
    loc = downloader.download(name_of_embedding=embed_name)
    if not downloader.is_converted(embed_name):
        downloader.extract(embed_name, loc, vec_size=vec_size)  # only one of the files
    vocab, vecs = downloader.load(embed_name)

    return vocab, vecs

    # downloader.convert(data)

"""

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

"""

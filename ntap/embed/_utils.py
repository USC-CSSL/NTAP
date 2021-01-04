import io
import zipfile
import os
import tempfile
import logging

from ntap.utils import download_file

def load_fasttext_vectors(fname):

    base_name = os.path.basename(fname)

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = np.zeros((n, d))
    vocab = [""] * n
    for i, line in tqdm(enumerate(fin), 
                        total=n, 
                        desc=f'convert_{base_name}'):
        tokens = line.rstrip().split(' ')
        vocab[i] = tokens[0]
        data[i, :] = np.array([float(t) for t in tokens[1:]])
    return vocab, data

def load_glove_vectors(fname, vocab_size, vec_size):

    base_name = os.path.basename(fname)

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = np.zeros((vocab_size, vec_size))
    vocab = [""] * vocab_size
    for i, line in tqdm(enumerate(fin), 
                        total=vocab_size, 
                        desc=f'convert_{base_name}'):
        tokens = line.rstrip().split(' ')
        vocab[i] = tokens[0]
        data[i, :] = np.array([float(t) for t in tokens[1:]])
    return vocab, data


class EmbeddingLoader:

    links = {
        'glove-wiki': "http://nlp.stanford.edu/data/glove.6B.zip",
        'glove-cc': "http://nlp.stanford.edu/data/glove.42B.300d.zip",
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
        'glove-cc': "glove.42B.300d.txt",
        'fasttext-wiki': "wiki-news-300d-1M.vec",
        'fasttext-wiki-subword': "wiki-news-300d-1M-subword.vec",
        'fasttext-cc': "crawl-300d-2M.vec",
        'fasttext-cc-subword': "crawl-300d-2M-subword.vec"
    }

    converters = {
        'fasttext': load_fasttext_vectors,
        'glove': load_glove_vectors
    }

    def __init__(self, base_dir):

        self.base_dir = os.path.expanduser(base_dir)

        self.base_dir = os.environ.get('NTAP_DATA_DIR', self.base_dir)
        self.log = logging.getLogger(__name__)

        if not os.path.isdir(self.base_dir):
            try:
                self.log.info(f'Creating {self.base_dir}')
                os.makedirs(self.base_dir)
            except OSError as e:
                self.log.exception("Can't create {}. ".format(self.base_dir))


    def extract(self, name, path_to_embedding_zip, vec_size=None):

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

                    if 'glove' in name:
                        assert vec_size is not None, "No vec size for GloVe"
                        glove_type = name.replace('glove-', '')
                        vocab_size = {'cc': 1917494, 'wiki': 400000}[glove_type]
                        vocab, data = converter(tmp_vec_path,
                                                vocab_size=vocab_size,
                                                vec_size=vec_size)
                    else:
                        vocab, data = converter(tmp_vec_path)

                local_output_dir = os.path.join(self.base_dir, name, str(vec_size))
                if not os.path.isdir(local_output_dir):
                    self.log.info(f"Creating {local_output_dir}")
                    os.makedirs(local_output_dir)
                local_vocab_file = os.path.join(local_output_dir, "vocab.txt")
                with open(local_vocab_file, 'w') as fo:
                    self.log.info(f"Writing vocab file at {local_vocab_file}")
                    fo.write('\n'.join(str(w) for w in vocab))

                local_vec_file = os.path.join(local_output_dir, 'vecs.npy')
                self.log.info(f"Writing {name}-{vec_size} vectors at {local_vec_file}")
                np.save(local_vec_file, data)

    def is_converted(self, name, vec_size):

        vocab_file = os.path.join(self.base_dir, name, str(vec_size), "vocab.txt")
        vec_file = os.path.join(self.base_dir, name, str(vec_size), "vecs.npy")

        return os.path.exists(vocab_file) and os.path.exists(vec_file)

    def load(self, name, vec_size):

        vocab_file = os.path.join(self.base_dir, name, str(vec_size), "vocab.txt")
        vec_file = os.path.join(self.base_dir, name, str(vec_size), "vecs.npy")
        with open(vocab_file, 'r') as fo:
            vocab = [line.strip() for line in fo if line.strip() != ""]
        vecs = np.load(vec_file)

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

        if not os.path.exists(local_download_loc):
            self.log.info(f'Downloading {name_of_embedding}')
            url = self.links[name_of_embedding]
            download_file(url, local_download_loc)
        return local_download_loc


def print_download_info(base_dir="~/ntap_data"):
    """ Go to base_dir and print downloaded/downloadable files """

    embed_downloader = EmbeddingLoader(base_dir)


    #dic_downloader = DictionaryLoader(base_dir)


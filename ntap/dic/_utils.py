from collections import Counter, defaultdict
import os
import re
import tempfile
import zipfile
import logging

import liwc

from ntap.utils import download_file

class DictionaryLoader:

    links = {
        'mfd': "https://moralfoundations.org/wp-content/uploads/files/downloads/moral%20foundations%20dictionary.dic"
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


        self.dic_dir = os.path.join(self.base_dir, 'dictionaries')
        if not os.path.isdir(self.dic_dir):
            self.log.info(f"Creating {self.dic_dir}")
            os.makedirs(self.dic_dir)

    def download(self, dic_name):

        if dic_name not in self.links:
            raise ValueError(f'{dic_name} not available')

        download_loc = self.links[dic_name]
        local_download_loc = os.path.join(self.dic_dir, dic_name + '.dic')

        if not os.path.exists(local_download_loc):
            self.log.info(f'Downloading {dic_name}')
            url = self.links[dic_name]
            download_file(url, local_download_loc)

        # some dic files are messy...
        with open(local_download_loc, 'r') as dic:
            lines = [l.strip() for l in dic if len(l.strip()) > 0]
            lines = [re.sub(r'[\t\s]+', '\t', l) for l in lines]
        with open(local_download_loc, 'w') as dic:
            dic.write('\n'.join(lines))

        return local_download_loc

    def load(self, dic_path):
        lexicon, _ = liwc.read_dic(dic_path)

        cat2lexicon = defaultdict(list)

        for word, categories in lexicon.items():
            for c in categories:
                if word not in cat2lexicon[c]:
                    cat2lexicon[c].append(word)

        self.lexicon = cat2lexicon
        return liwc.load_token_parser(dic_path)

    """
    def info(self, print_downloaded=True, print_available=True):
        if print_downloaded:
            downloaded_embeddings = list()
            for dir_ in os.listdir(self.base_dir):
                pass
    """

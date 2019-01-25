"""
Apply Entity Linking to a tweet dataset using TagMe's RESTful API.
"""

import time
import requests

from argparse import ArgumentParser

arg_parser = ArgumentParser()
arg_parser.add_argument('-t', '--token', help='TagMe token')
arg_parser.add_argument('-i', '--input', help='Path to the input file')
arg_parser.add_argument('-o', '--output', help='Path to the output file')

TAGME_URL = 'https://tagme.d4science.org/tagme/tag'

# Read TagMe token
# dirname = os.path.dirname(__file__)
# with open(os.path.join(dirname, 'tagme_token'), 'r', encoding='utf-8') as r:
#     TAGME_TOKEN = r.readline().strip()

DEFAULT_OPTIONS = {
    'lang': 'en',
    # 'gcube-token': TAGME_TOKEN,
    'tweet': 'true',
    'include_abstract': 'true',
    'include_categories': 'true'
}

def tagme(input_file, output_file, token):
    with open(input_file, 'r', encoding='utf-8') as r, \
        open(output_file, 'w', encoding='utf-8') as w:
        for line_idx, line in enumerate(r, 1):
            tid, text, _labels = line.strip().split('\t')
            params = {
                'text': text,
                'gcube-token': token,
                **DEFAULT_OPTIONS
            }
            res = requests.get(TAGME_URL, params=params)
            if res.status_code == 200:
                w.write('{}\t{}\n'.format(tid, res.text.replace('\n', ' ')))
            else:
                print('failed to link {}, {}'.format(line_idx, tid))
            if line_idx % 100 == 0:
                print('[{}] sleep for 1 minute'.format(line_idx))
                time.sleep(60)


if __name__ == '__main__':
    args = arg_parser.parse_args()
    tagme(args.input, args.output, args.token)
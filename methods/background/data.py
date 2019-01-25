import torch
from torch.autograd import Variable
import nltk
from nltk import TweetTokenizer, word_tokenize, pos_tag
from collections import namedtuple, Counter, defaultdict
from random import shuffle, choice
# from numpy.random import choice
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNK = '$UNK$'
TweetInstance = namedtuple('TweetInstance', ['tid', 'text', 'labels'])
TagmeInstance = namedtuple('TagmeInstance', ['tid', 'result'])
tokenizer = TweetTokenizer(reduce_len=False)


nltk.download('punkt')

class Dataset:
    def __init__(self, path, labels):
        self.path = path
        self.tweet_count = 0
        self.raw_dataset = []
        self.nbz_dataset = []
        self.dataset = []
        self.token_vocab = {UNK: 0}
        # labels
        self.label_vocab = {l: i for i, l in
                            enumerate(['NM'] + labels)}

        self.load_dataset()
        self.data_stats()

    def load_dataset(self, skip_first=False):
        """Load raw data from file.

        :param skip_first: Skip the first line (default=False). 
        """
        logger.info('Loading data from {}'.format(self.path))
        with open(self.path, 'r', encoding='utf-8') as r:
            if skip_first:
                next(r)
            for line in r:
                tid, text, labels = line.rstrip().split('\t')
                text = text.strip()
                if len(text) == 0:
                    logger.warning('Skipped an empty tweet {}'.format(tid))
                    continue
                labels = [l.strip() for l in labels.split(',')]
                self.raw_dataset.append(TweetInstance(tid, text, labels))
        self.tweet_count = len(self.raw_dataset)

    def data_stats(self):
        token_count = Counter()
        label_count = Counter()
        for tid, text, labels in self.raw_dataset:
            tokens = tokenizer.tokenize(text)
            tokens_lower = [t.lower() for t in tokens]
            token_count.update(tokens_lower)
            label_count.update(labels)
        return token_count, label_count

    def numberize_dataset(self):
        """Numberize the dataset."""
        for tid, text, labels in self.raw_dataset:
            tokens = [t.lower() for t in tokenizer.tokenize(text)]
            tokens_nbz = [self.token_vocab[t] if t in self.token_vocab else 0
                          for t in tokens]
            labels_nbz = [self.label_vocab[l] for l in labels]
            self.nbz_dataset.append(TweetInstance(tid, tokens_nbz, labels_nbz))

    def shuffle_dataset(self, label, balance=True):
        """Shuffle and balance the data set.

        :param label: Target label.
        :param balance: Balance positive and negative instances.
        """
        self.dataset = []

        label_index = self.label_vocab[label]

        positives = []
        negatives = []
        for tid, tokens, labels in self.nbz_dataset:
            if label_index in labels:
                positives.append(TweetInstance(tid, tokens, 1))
            else:
                negatives.append(TweetInstance(tid, tokens, 0))

        self.dataset = positives + negatives
        if balance:
            pos_num = len(positives)
            neg_num = len(negatives)
            if pos_num > neg_num:
                self.dataset += [choice(negatives)
                                 for _ in range(pos_num - neg_num)]
            elif neg_num > pos_num:
                self.dataset += [choice(positives)
                                 for _ in range(neg_num - pos_num)]
        shuffle(self.dataset)

    def init_dataset(self, label):
        self.dataset = []

        label_index = self.label_vocab[label]

        for tid, tokens, labels in self.nbz_dataset:
            self.dataset.append(
                TweetInstance(tid, tokens, label_index in labels))

    def get_batch(self, batch_size=10, max_len=30, volatile=False, gpu=False):
        batch_tids = []
        batch_tokens = []
        batch_labels = []
        batch_lens = []
        for i in range(batch_size):
            tid, tokens, label = self.dataset.pop(0)
            tokens = tokens[:max_len]
            batch_lens.append(len(tokens))
            if len(tokens) < max_len:
                tokens += [0] * (max_len - len(tokens))
            batch_tids.append(tid)
            batch_tokens.append(tokens)
            batch_labels.append(label)
        batch_lens, batch_tids, batch_tokens, batch_labels = zip(*sorted(zip(
            batch_lens, batch_tids, batch_tokens, batch_labels), reverse=True))

        batch_lens = Variable(torch.LongTensor(batch_lens), volatile=volatile)
        batch_tokens = Variable(torch.LongTensor(batch_tokens),
                                volatile=volatile)
        batch_labels = Variable(torch.LongTensor(batch_labels),
                                volatile=volatile)

        if gpu:
            batch_lens = batch_lens.cuda()
            batch_labels = batch_labels.cuda()
            # TODO: Look up embeddings on CPU
            batch_tokens = batch_tokens.cuda()

        return batch_tids, batch_tokens, batch_labels, batch_lens

    def get_dataset(self, max_len, volatile=False, gpu=False):
        batch_size = len(self.dataset)
        return self.get_batch(batch_size, max_len, volatile=volatile, gpu=gpu)

    def batch_num(self, batch_size=10):
        return len(self.dataset) // batch_size


class TagMeResult:
    def __init__(self, path):
        self.path = path
        self.raw_dataset = []
        self.tid_vector = {}
        self.load()

    def load(self):
        with open(self.path, 'r', encoding='utf-8') as r:
            for line in r:
                tid, result = line.strip().split('\t')
                result = json.loads(result)
                self.raw_dataset.append(TagmeInstance(tid=tid, result=result))

    def numberize_dataset(self, embedding_file, embedding_dim, threshold=.1):
        embeddings = {}
        logger.info('Load embeddings from {}'.format(embedding_file))
        with open(embedding_file, 'r', encoding='utf-8') as r:
            for line in r:
                line = line.rstrip().split(' ')
                embeddings[line[0]] = torch.FloatTensor(
                    [float(v) for v in line[1:]])

        top_abstract = {}
        for _tid, result in self.raw_dataset:
            for anno in result['annotations']:
                if 'title' in anno and 'abstract' in anno:
                    spot = anno['spot'].lower()
                    rho = anno['rho']
                    if rho < threshold:
                        continue
                    if spot not in top_abstract or top_abstract[spot][0] < rho:
                        top_abstract[spot] = (rho, anno['abstract'])

        for tid, result in self.raw_dataset:
            knowledge_vector = torch.FloatTensor(embedding_dim).fill_(0)
            token_num_total = 0
            for anno in result['annotations']:
                if 'title' in anno and 'abstract' in anno:
                    abstract = anno['abstract']
                    spot = anno['spot'].lower()
                    if anno['rho'] < threshold:
                        if spot in top_abstract:
                            abstract = top_abstract[spot][1]
                        else:
                            continue
                    vector = torch.FloatTensor(embedding_dim).fill_(0)
                    token_num = 0

                    tokens = [w.lower() for w in word_tokenize(abstract)]
                    for t in tokens:
                        if t in embeddings:
                            vector += embeddings[t]
                            token_num += 1
                    if token_num > 0:
                        token_num_total += token_num
                        vector = vector.div(token_num)
                        knowledge_vector += vector * anno['rho']
            if token_num_total > 0:
                knowledge_vector = knowledge_vector.div(
                    knowledge_vector.norm(p=2))
            self.tid_vector[tid] = knowledge_vector

    def get_batch(self, tids, volatile=False, gpu=False):
        vectors = []
        for tid in tids:
            vectors.append(self.tid_vector[tid])
        vectors = torch.stack(vectors)
        vectors = Variable(vectors, volatile=volatile)
        if gpu:
            vectors = vectors.cuda()
        return vectors


class MfdResult:
    def __init__(self, data_path, dict_path):
        self.data_path = data_path
        self.dict_path = dict_path
        self.category_num = 0
        self.token_category = defaultdict(list)
        self.tid_vector = {}
        self.load()

    def load(self):
        # Load dictionary
        inBody = False
        with open(self.dict_path, 'r', encoding='utf-8') as r:
            next(r)
            for line in r:
                if inBody:
                    segs = line.strip().split('\t')
                    token = segs[0]
                    for cate_id in segs[1:]:
                        self.token_category[token].append(int(cate_id))
                else:
                    if line.startswith('%'):
                        inBody = True
                    else:
                        self.category_num += 1
        tokenizer = TweetTokenizer()
        with open(self.data_path, 'r', encoding='utf-8') as r:
            for line in r:
                tid, tweet, _ = line.rstrip().split('\t')
                tokens = tokenizer.tokenize(tweet)
                tokens = [t.replace('#', '').lower() for t in tokens]
                category_count = [0] * self.category_num
                for token in tokens:
                    for i in range(min(len(token), 5)):
                        if token[:-i] in self.token_category:
                            for cate in self.token_category[token[:-i]]:
                                category_count[cate - 1] += 1
                            break
                if len(tokens) > 0:
                    category_count = [c / len(tokens) for c in category_count]
                self.tid_vector[tid] = torch.FloatTensor(category_count)

    def numberize_dataset(self):
        pass

    def get_batch(self, tids, volatile=False, gpu=False):
        vectors = []
        for tid in tids:
            vectors.append(self.tid_vector[
                               tid] if tid in self.tid_vector else torch.FloatTensor(
                self.category_num))
        vectors = torch.stack(vectors)
        vectors = Variable(vectors, volatile=volatile)
        if gpu:
            vectors = vectors.cuda()
        return vectors

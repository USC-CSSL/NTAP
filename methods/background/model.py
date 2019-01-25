"""
Code Repurposed from Ying Lin:
    Ling et al., 2018:
        @inproceedings{lin2018acquiring,
          title={Acquiring background knowledge to improve moral value prediction},
          author={Lin, Ying and Hoover, Joe and Portillo-Wightman, Gwenyth and Park, Christina and Dehghani, Morteza and Ji, Heng},
          booktitle={2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)},
          pages={552--559},
          year={2018},
          organization={IEEE}
    }
"""



import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.utils.rnn as R
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


class Linear(nn.Linear):
    """
    Linear Layer.
    Add the initialize function.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True):
        super(Linear, self).__init__(in_features,
                                     out_features,
                                     bias=bias)
        self.initialize()

    def initialize(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                I.orthogonal(p)
            elif 'bias' in n:
                I.uniform(p, b=.1)


class Embedding(nn.Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 pretrain=None,
                 vocab=None,
                 trainable=False,
                 stats=True
                 ):
        super(Embedding, self).__init__(num_embeddings,
                                        embedding_dim,
                                        padding_idx,
                                        max_norm,
                                        norm_type,
                                        scale_grad_by_freq,
                                        sparse)
        self.output_size = embedding_dim
        self.num_embeddings = num_embeddings
        self.pretrain = pretrain
        self.vocab = vocab
        self.stats = stats
        if not trainable:
            self.weight.requires_grad = False
        if pretrain and vocab:
            self.load(pretrain, vocab, stats=stats)
        else:
            I.xavier_normal(self.weight.data)

    def load(self, path, vocab, stats=False):
        """Load pre-trained embeddings from file.
        Only supports text format embedding and

        :param path: Path to the embedding file.
        :param vocab: Vocab dict.
        :param stats: Is the first line stats info.
        :return: 
        """
        logger.info('Loading word embeddings from {}'.format(path))
        with open(path, 'r', encoding='utf-8', errors='ignore') as r:
            if stats:
                r.readline()
            for line in r:
                line = line.rstrip().split(' ')
                token = line[0]
                if token in vocab:
                    vector = self.weight.data.new([float(v) for v in line[1:]])
                    self.weight.data[vocab[token]] = vector


class LSTM(nn.LSTM):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0,
                 bidirectional=False,
                 forget_bias=0):
        super(LSTM, self).__init__(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bias=bias,
                                   batch_first=batch_first,
                                   dropout=dropout,
                                   bidirectional=bidirectional)
        self.forget_bias = forget_bias
        self.output_size = hidden_size * (2 if bidirectional else 1)
        self.bidirectional = bidirectional
        self.initialize()

    def initialize(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                # I.xavier_normal(p)
                I.orthogonal(p)
            elif 'bias' in n:
                bias_size = p.size(0)
                p.data[bias_size // 4:bias_size // 2].fill_(self.forget_bias)

    def forward(self, inputs, lens, hx=None):
        inputs_packed = R.pack_padded_sequence(inputs, lens.data.tolist(),
                                               batch_first=True)
        outputs, h = super(LSTM, self).forward(inputs_packed, hx)
        outputs, _ = R.pad_packed_sequence(outputs, batch_first=True)
        return outputs, h


class MoralClassifier(nn.Module):
    def __init__(self,
                 word_embedding,
                 lstm,
                 linears,
                 embed_dropout_prob=.5,
                 lstm_dropout_prob=.5,
                 gpu=False):
        super(MoralClassifier, self).__init__()

        self.word_embedding = word_embedding
        self.lstm = lstm
        self.linears = nn.ModuleList(linears)
        self.linear_num = len(linears)
        self.embed_dropout = nn.Dropout(p=embed_dropout_prob)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout_prob)
        self.gpu = gpu

    def forward(self, tokens, lens):
        # embedding lookup
        tokens_embed = self.word_embedding.forward(tokens)
        tokens_embed = self.embed_dropout.forward(tokens_embed)

        # lstm layer
        _lstm_outputs, (last_hidden, _last_cell) = self.lstm.forward(
            tokens_embed, lens)
        last_hidden = last_hidden.squeeze(0)
        last_hidden = self.lstm_dropout.forward(last_hidden)

        # linear layers
        linear_input = last_hidden
        for layer_idx, linear in enumerate(self.linears):
            linear_input = linear.forward(linear_input)
            # if layer_idx != self.linear_num - 1:
            #     linear_input = F.dropout(linear_input, p=.2)
        return linear_input


class MoralClassifierExt(nn.Module):
    def __init__(self,
                 word_embedding,
                 lstm,
                 linears,
                 ext_linears,
                 embed_dropout_prob=.5,
                 lstm_dropout_prob=.5,
                 el_dropout_prob=.5,
                 gpu=False):
        super(MoralClassifierExt, self).__init__()

        self.word_embedding = word_embedding
        self.lstm = lstm
        self.linears = nn.ModuleList(linears)
        self.ext_linears = nn.ModuleList(ext_linears)
        self.linear_num = len(linears)
        self.ext_linear_num = len(ext_linears)
        self.embed_dropout = nn.Dropout(p=embed_dropout_prob)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout_prob)
        self.el_dropout = nn.Dropout(p=el_dropout_prob)
        self.gpu = gpu

    def forward(self, tokens, lens, exts):
        # TODO: add non-linear functions
        # embedding lookup
        tokens_embed = self.word_embedding.forward(tokens)
        tokens_embed = self.embed_dropout.forward(tokens_embed)

        # lstm layer
        _lstm_outputs, (last_hidden, _last_cell) = self.lstm.forward(
            tokens_embed, lens)
        last_hidden = last_hidden.squeeze(0)
        last_hidden = self.lstm_dropout.forward(last_hidden)

        # ext linear layers
        ext_linear_input = exts
        for layer_idx, linear in enumerate(self.ext_linears):
            ext_linear_input = linear.forward(ext_linear_input)
        ext_linear_input = F.relu(ext_linear_input)
        ext_linear_input = self.el_dropout.forward(ext_linear_input)

        # linear layers
        linear_input = torch.cat([last_hidden, ext_linear_input], dim=1)
        for layer_idx, linear in enumerate(self.linears):
            linear_input = linear.forward(linear_input)

        return linear_input


class MoralClassifierMfdBk(nn.Module):
    def __init__(self,
                 word_embedding,
                 lstm,
                 linears,
                 el_linears,
                 mfd_linears,
                 embed_dropout_prob=.5,
                 lstm_dropout_prob=.5,
                 el_dropout_prob=.5,
                 mfd_dropout_prob=.5,
                 gpu=False):
        super(MoralClassifierMfdBk, self).__init__()

        self.word_embedding = word_embedding
        self.lstm = lstm
        self.linears = nn.ModuleList(linears)
        self.el_linears = nn.ModuleList(el_linears)
        self.mfd_linears = nn.ModuleList(mfd_linears)
        self.linear_num = len(linears)
        self.el_linear_num = len(el_linears)
        self.mfd_linear_num = len(mfd_linears)
        self.embed_dropout = nn.Dropout(p=embed_dropout_prob)
        self.lstm_dropout = nn.Dropout(p=lstm_dropout_prob)
        self.el_dropout = nn.Dropout(p=el_dropout_prob)
        self.mfd_dropout = nn.Dropout(p=mfd_dropout_prob)
        self.gpu = gpu

    def forward(self, tokens, lens, els, mfds):
        # TODO: add non-linear functions
        # embedding lookup
        tokens_embed = self.word_embedding.forward(tokens)
        tokens_embed = self.embed_dropout.forward(tokens_embed)

        # lstm layer
        _lstm_outputs, (last_hidden, _last_cell) = self.lstm.forward(
            tokens_embed, lens)
        last_hidden = last_hidden.squeeze(0)
        last_hidden = self.lstm_dropout.forward(last_hidden)

        # el linear layers
        el_linear_input = els
        for layer_idx, linear in enumerate(self.el_linears):
            el_linear_input = linear.forward(el_linear_input)
        el_linear_input = F.relu(el_linear_input)
        el_linear_input = self.el_dropout.forward(el_linear_input)

        # mfd linear layers
        mfd_linear_input = mfds
        for layer_idx, linear in enumerate(self.mfd_linears):
            mfd_linear_input = linear.forward(mfd_linear_input)
        mfd_linear_input = F.relu(mfd_linear_input)
        mfd_linear_input = self.el_dropout.forward(mfd_linear_input)

        # linear layers
        linear_input = torch.cat(
            [last_hidden, el_linear_input, mfd_linear_input], dim=1)
        for layer_idx, linear in enumerate(self.linears):
            linear_input = linear.forward(linear_input)

        return linear_input

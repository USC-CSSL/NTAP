import logging

FORMAT = '%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

# Keep
from model import MoralClassifierMfdBk, Embedding, LSTM, Linear
# want to do without Dataset, TagMeResult, MfdResult...
from data import Dataset, TagMeResult, MfdResult
from argparse import ArgumentParser
from collections import namedtuple, defaultdict

import os
import time
import torch

#current_time = lambda: int(round(time.time() * 1000))
Scores = namedtuple('Scores',
                    ['tp', 'tn', 'fp', 'fn', 'recall', 'precision', 'fscore'])
arg_parser = ArgumentParser()
arg_parser.add_argument('--labels', default='CH,FC,AS,LB,PD', help='Label set')
arg_parser.add_argument('--gpu', default=1, type=int, help='Use GPU')
arg_parser.add_argument('--device', type=int, help='Selece GPU')

# from utils import load_glove, load_dict, load_word2vec

class BackgroundKnowledge:
    def __init__(self, source_file, 
                       tagme_dir,
                       dest_dir,
                       params):
        if type(params) == dict:
            self.params = params
        else:
            with open(params, 'r') as fo:
                self.params = json.load(fo)
        self.model_dir = dest_dir
        self.data_dir = 
        self.tagme_dir = os.path.join(source_dir, "tagme")
        self.domain_dict = load_dict(self.params["dictionary"])
        embedding_file = args.embedding  # GLOVE? 
        el_embedding_file = args.el_embedding  # need to find out why there are EL embed

        embedding_dim = args.embedding_dim
        el_embedding_dim = args.el_embedding_dim
        if type(self.params["linears"]) != list:
            self.params["linears"] = [self.params["linears"]]
        if type(self.params["dict_linears"]) != list:
            self.params["dict_linears"] = [self.params["dict_linears"]]
        if type(self.params["el_linears"] = [self.params["el_linears"]]
        
        self.use_gpu = False
        """
        use_gpu = args.gpu > 0 and torch.cuda.device_count() > 0
        if use_gpu and args.device:
            print('Select GPU {}'.format(args.device))
            torch.cuda.set_device(args.device)
        """

    def init_data(self)
        train_set = Dataset(train_file, labels=labels)
        dev_set = Dataset(dev_file, labels=labels)
        test_set = Dataset(test_file, labels=labels)
        el_set = TagMeResult(el_file)
        train_mfd_set = MfdResult(train_file, mfd_file)
        dev_mfd_set = MfdResult(dev_file, mfd_file)
        test_mfd_set = MfdResult(test_file, mfd_file)

        train_token_count, train_label_count = train_set.data_stats()
        dev_token_count, dev_label_count = dev_set.data_stats()
        test_token_count, test_label_count = test_set.data_stats()

        token_vocab = {'$UNK$': 0}
        label_vocab = {'NM': 0}

        for t in list(train_token_count.keys()) \
                + list(dev_token_count.keys()) + list(test_token_count.keys()):
            if t not in token_vocab:
                token_vocab[t] = len(token_vocab)
        for l in list(train_label_count.keys()) \
                + list(dev_label_count.keys()) + list(test_label_count.keys()):
            if l not in label_vocab:
                label_vocab[l] = len(label_vocab)

        train_set.token_vocab = token_vocab
        dev_set.token_vocab = token_vocab
        test_set.token_vocab = token_vocab
        train_set.label_vocab = label_vocab
        dev_set.label_vocab = label_vocab
        test_set.label_vocab = label_vocab

        train_set.numberize_dataset()
        dev_set.numberize_dataset()
        test_set.numberize_dataset()

        el_set.numberize_dataset(el_embedding_file, el_embedding_dim)

    def build(self):
        models = {}
        optimizers = {}
        word_embedding = Embedding(len(token_vocab),
                                   embedding_dim,
                                   padding_idx=0,
                                   sparse=True,
                                   pretrain=embedding_file,
                                   vocab=token_vocab,
                                   trainable=True)
        lstm = LSTM(embedding_dim, hidden_size, batch_first=True, forget_bias=1.0)

        linear1 = [self.h + self.el_linears[-1] + self.dict_linears[-1]] + self.linears
        linear2 = self.linears + [2]  # 2 because binary output
        linears = [Linear(i, 0) for i, o in zip(linear1, linear2)]

        el_1 = [self.params["embeddingdim"]] + self.params["el_linears"][:-1]
        el_2 = self.params["el_linears"]
        el_linears = [Linear(i, o) for i, o in zip(el_1, el_2)]

        dict_1 = [(size of dictionary)] + self.params["dict_linears"][:-1]
        dict_2 = self.params["dict_linears"]
        dict_linears = [Linear(i, o) for i, o in zip(dict_1, dict_2)]

        model = MoralClassifierMfdBk(word_embedding, lstm, 
                                     linears, el_linears, dict_linears)
        if self.use_gpu:
            model.cuda()

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate, momentum=.9)
        
        self.model = model
        self.optimizer = optimizer
        self.loss_func = torch.nn.CrossEntropyLoss()

    def train(self):

        # ----------------------------------------------------------------------
        # Mode: train
        test_label_preds = {}
        best_scores = {}
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        for target_label in labels:
            model_file = os.path.join(model_dir,
                                      'checkpoint_{}.mdl'.format(target_label))
            model = models[target_label]
            optimizer = optimizers[target_label]
            # TODO: combine init_dataset() and shuffle_dataset()
            dev_set.init_dataset(target_label)
            test_set.init_dataset(target_label)
            (
                dev_tids, dev_tokens, dev_labels, dev_lens
            ) = dev_set.get_dataset(max_seq_len, volatile=True, gpu=use_gpu)
            (
                test_tids, test_tokens, test_labels, test_lens
            ) = test_set.get_dataset(max_seq_len, volatile=True, gpu=use_gpu)
            test_el = el_set.get_batch(test_tids, volatile=True, gpu=use_gpu)
            dev_el = el_set.get_batch(dev_tids, volatile=True, gpu=use_gpu)
            test_mfd = test_mfd_set.get_batch(test_tids, volatile=True, gpu=use_gpu)
            dev_mfd = dev_mfd_set.get_batch(dev_tids, volatile=True, gpu=use_gpu)

            best_dev_fscore = 0.0
            best_test_scores = None
            for epoch in range(max_epoch):
                epoch_start_time = current_time()
                epoch_loss = 0.0
                train_set.shuffle_dataset(target_label, balance=True)
                batch_num = train_set.batch_num(batch_size)
                for batch_idx in range(batch_num):
                    optimizer.zero_grad()
                    (
                        batch_tids, batch_tokens, batch_labels, batch_lens
                    ) = train_set.get_batch(batch_size, gpu=use_gpu)
                    batch_el = el_set.get_batch(batch_tids,
                                                volatile=False, gpu=use_gpu)
                    batch_mfd = train_mfd_set.get_batch(batch_tids,
                                                volatile=False, gpu=use_gpu)
                    model_output = model.forward(batch_tokens, batch_lens, batch_el, batch_mfd)
                    loss = loss_func.forward(model_output, batch_labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += 1.0 / batch_num * float(loss)
                epoch_elapsed_time = current_time() - epoch_start_time

                # Evaluate the current model on dev and test sets
                dev_preds = model.forward(dev_tokens, dev_lens, dev_el, dev_mfd)
                dev_scores = _calc_scores(dev_preds, dev_labels)
                test_preds = model.forward(test_tokens, test_lens, test_el, test_mfd)
                test_scores = _calc_scores(test_preds, test_labels)
                # Output score
                logger.info('[{}] Epoch {:<3} [{}ms]: {:.4f} | P: {:<5.2f} '
                            'R: {:<5.2f} F: {:<5.2f}{}'.format(
                    target_label, epoch, epoch_elapsed_time, epoch_loss,
                    dev_scores.precision, dev_scores.recall, dev_scores.fscore,
                    ' *' if dev_scores.fscore > best_dev_fscore else ''
                ))
                # Save the best model based on performance on the dev set
                if dev_scores.fscore > best_dev_fscore:
                    best_dev_fscore = dev_scores.fscore
                    states_to_save = {
                        'token_vocab': token_vocab,
                        'label_vocab': label_vocab,
                        'model': model.state_dict(),
                        'embedding_dim': embedding_dim,
                        'hidden_size': hidden_size,
                        'linear_sizes': linear_sizes,
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(states_to_save, model_file)
                    best_test_scores = test_scores
                    test_label_preds[target_label] = (test_tids, test_preds)
            if best_test_scores:
                logger.info('Label: {}'.format(target_label))
                logger.info('Precision: {:.2f}, Recall: {:.2f}, F-score: {:.2f}'.format(
                    best_test_scores.precision,
                    best_test_scores.recall,
                    best_test_scores.fscore
                ))
                best_scores[target_label] = best_test_scores
            print('-' * 80)
        nm_scores = _calc_non_moral_scores(test_label_preds, test_set)
        logger.info('Label: NM')
        logger.info('Precision: {:.2f}, Recall: {:.2f}, F-score: {:.2f}'.format(
            nm_scores.precision, nm_scores.recall, nm_scores.fscore
        ))
        print('-' * 80)
        print(best_scores)
    def _calc_scores(prediction, gold):
        _, pred_idx = torch.max(prediction, dim=1)
        tp = int(sum((pred_idx == 1) * (gold == 1)))
        tn = int(sum((pred_idx == 0) * (gold == 0)))
        fp = int(sum((pred_idx == 1) * (gold == 0)))
        fn = int(sum((pred_idx == 0) * (gold == 1)))
        recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        fscore = 0 if (recall == 0 or precision == 0) \
            else 2.0 * (precision * recall) / (precision + recall)
        return Scores(tp=tp, tn=tn, fp=fp, fn=fn,
                      recall=recall * 100,
                      precision=precision * 100,
                      fscore=fscore * 100)

    def _calc_non_moral_scores(label_preds, ds):
        ds.init_dataset('NM')
        gold = {}
        for tid, tokens, label in ds.dataset:
            gold[tid] = label

        tid_preds = defaultdict(list)
        for label, (tids, preds) in label_preds.items():
            _, preds = torch.max(preds, dim=1)
            preds = preds.data.tolist()
            for tid, pred in zip(tids, preds):
                tid_preds[tid].append(pred)
        tid_preds = {k: sum(v) == 0 for k, v in tid_preds.items()}

        tp = tn = fp = fn = 0
        for tid, p in tid_preds.items():
            g = gold[tid]
            if p and g:
                tp += 1
            elif p and not g:
                fp += 1
            elif not p and g:
                fn += 1
            else:
                tn += 1
        recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        fscore = 0 if (recall == 0 or precision == 0) \
            else 2.0 * (precision * recall) / (precision + recall)
        return Scores(tp=tp, tn=tn, fp=fp, fn=fn,
                      recall=recall * 100,
                      precision=precision * 100,
                      fscore=fscore * 100)


    def _gen_result_file(label_preds, data_file, result_file):
        # Gold labels
        tid_gold = {}
        with open(data_file, 'r', encoding='utf-8') as r:
            for line in r:
                tid, text, labels = line.rstrip().split('\t')
                tid_gold[tid] = labels

        tid_preds = defaultdict(list)
        for label, (tids, preds) in label_preds.items():
            _, preds = torch.max(preds, dim=1)
            preds = preds.data.tolist()
            for tid, pred in zip(tids, preds):
                if pred == 1:
                    tid_preds[tid].append(label)

        with open(result_file, 'w', encoding='utf-8') as w:
            for tid, gold in tid_gold.items():
                if tid in tid_preds:
                    w.write('{}\t{}\t{}\n'.format(tid, gold, ','.join(tid_preds[tid])))
                else:
                    w.write('{}\t{}\t{}\n'.format(tid, gold, 'NM'))

#_gen_result_file(test_label_preds, test_file, output_file)

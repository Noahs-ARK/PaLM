import os
import torch, re
import nltk
from nltk.corpus import ptb
from collections import Counter
from utils import get_brackets
word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB', "$", "#"]
currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']


P = re.compile("[-+]?\d*\.\d+|[-+]\d+|[-+]?\d*\,\d+|\d+|\d+:\d+")
PY = re.compile("\d+[%]?-[a-zA-Z]|\d+[%]?[a-zA-Z]|[-+]?\d*\.\d+[%]?-[a-zA-Z]|[-+]?\d*\.\d+[%]?[a-zA-Z]|[-+]?\d*\,\d+-[%]?[a-zA-Z]|[-+]?\d*\,\d+[%]?[a-zA-Z]")
PA = re.compile("\d+-\d+-\d+")
PB = re.compile("\d+\\\/\d+-[A-Za-z]|\d+\\\/\d+[A-Za-z]")
L = ["a310-300s", "747-100s", "747-400s", "45,000-$60,000", "767-300er", "747-400s"]

file_ids = ptb.fileids()

train_file_ids = []
valid_file_ids = []
test_file_ids = []
rest_file_ids = []
train_lm_file_ids = []
for id in file_ids:
    if 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/24/WSJ_2499.MRG':
        train_file_ids.append(id)
    if 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/20/WSJ_2099.MRG':
        train_lm_file_ids.append(id)
    if 'WSJ/22/WSJ_2200.MRG' <= id <= 'WSJ/22/WSJ_2299.MRG':
        valid_file_ids.append(id)
    if 'WSJ/23/WSJ_2300.MRG' <= id <= 'WSJ/23/WSJ_2399.MRG':
        test_file_ids.append(id)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def id2tokens(self, ids):
        return [self.idx2word[w] for w in ids]

    def token2ids(self, sent):
        return [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"] for w in sent]

class Corpus(object):
    def __init__(self, path, max_span_length, dictionary=None):
        self.max_span_length = max_span_length
        if dictionary is None:
            self.dictionary = Dictionary()
        else:
            self.dictionary = dictionary
        self.train_sens, self.train_list_trees, _ = self.trees(train_lm_file_ids)
        self.train, self.train_trees = self.tokenize(os.path.join(path, 'train_ptb'), is_train=True)
        self.valid, _ = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test, _ = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path, is_train=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            trees = []
            n = 0
            for line in f:
                words = line.split() + ['<eos>']
                len_sent = len(words)
                tree = None
                if is_train:
                    if n == 0:
                        print (words)
                        tree = torch.zeros(len(words), self.max_span_length - 1).detach()
                    else:
                        assert n >= 1
                        tree = self.train_list_trees[n - 1]
                        brackets, _ = get_brackets(tree)

                        y_sent = []
                        y_sent.append([0] * (self.max_span_length - 1))
                        for i in range(len_sent - 1):
                            end_idx = i + 1
                            start_idx = max(0, end_idx - self.max_span_length + 1)
                            y_token = []
                            for k in range(start_idx, end_idx):
                                span = (k, end_idx)
                                s = self.max_span_length - (k - start_idx) - 1
                                if span in brackets:
                                    y_token.append(s)
                                else:
                                    y_token.append(0)
                            rem = max(0, self.max_span_length - len(y_token) - 1)
                            y_token = y_token + [0] * rem
                            assert len(y_token) == self.max_span_length - 1
                            y_sent.append(y_token)
                        assert len(y_sent) == len_sent
                        tree = torch.FloatTensor(y_sent).detach()
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
                trees.append(tree)
                n += 1
            trees = torch.cat(trees, dim=0) if is_train else None
        if is_train:
            assert len(self.train_list_trees) == (n - 1)
        return ids, trees

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            if tag in word_tags:
                w = w.lower()

                new_w = w
                if tag in ["CD", "LS", "JJ", "NNP", "RB"] and re.match(P, w) and not re.match(PY, w) and not re.match(PB, w) or w == "%":
                    new_w = "N"
                if w == "%":
                    new_w = "N"
                if re.match(PA, w):
                    new_w = "N"
                if re.match("\d+$", w):
                    new_w = "N"
                if w in L:
                    new_w = w

                words.append(new_w)

        return words

    def trees(self, file_ids):

        def tree2list(tree):
            if isinstance(tree, nltk.Tree):
                if tree.label() in word_tags:
                    w = tree.leaves()[0].lower()
                    w = re.sub('[0-9]+', 'N', w)
                    return w
                else:
                    root = []
                    for child in tree:
                        c = tree2list(child)
                        if c != []:
                            root.append(c)
                    if len(root) > 1:
                        return root
                    elif len(root) == 1:
                        return root[0]
            return []


        trees = []
        nltk_trees = []
        sens = []
        for id in file_ids:
            sentences = ptb.parsed_sents(id)

            for sen_tree in sentences:
                words = self.filter_words(sen_tree)
                words = words + ['<eos>']
                sens.append(words)
                nltk.treetransforms.chomsky_normal_form(sen_tree)
                trees.append(tree2list(sen_tree))
                nltk_trees.append(sen_tree)

        return sens, trees, nltk_trees

    def id2tokens(self, ids):
        return self.dictionary.id2tokens(ids)

    def token2ids(self, sents):
        return self.dictionary.token2ids(sents)

class WikiCorpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

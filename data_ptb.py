import os
import re
import pickle
import nltk
from nltk.corpus import ptb
word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']

file_ids = ptb.fileids()
train_file_ids = []
valid_file_ids = []
test_file_ids = []
rest_file_ids = []
train_lm_file_ids = []


P = re.compile("[-+]?\d*\.\d+|[-+]\d+|[-+]?\d*\,\d+|\d+|\d+:\d+")
PY = re.compile("\d+[%]?-[a-zA-Z]|\d+[%]?[a-zA-Z]|[-+]?\d*\.\d+[%]?-[a-zA-Z]|[-+]?\d*\.\d+[%]?[a-zA-Z]|[-+]?\d*\,\d+-[%]?[a-zA-Z]|[-+]?\d*\,\d+[%]?[a-zA-Z]")
PA = re.compile("\d+-\d+-\d+")
PB = re.compile("\d+\\\/\d+-[A-Za-z]|\d+\\\/\d+[A-Za-z]")
L = ["a310-300s", "747-100s", "747-400s", "45,000-$60,000", "767-300er", "747-400s"]
for id in file_ids:
    if 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/24/WSJ_2499.MRG':
        train_file_ids.append(id)
    if 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/20/WSJ_2099.MRG':
        train_lm_file_ids.append(id)
    if 'WSJ/22/WSJ_2200.MRG' <= id <= 'WSJ/22/WSJ_2299.MRG':
        valid_file_ids.append(id)
    if 'WSJ/23/WSJ_2300.MRG' <= id <= 'WSJ/23/WSJ_2399.MRG':
        test_file_ids.append(id)
    # elif 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/01/WSJ_0199.MRG' or 'WSJ/24/WSJ_2400.MRG' <= id <= 'WSJ/24/WSJ_2499.MRG':
    #     rest_file_ids.append(id)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.word2frq = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        if word not in self.word2frq:
            self.word2frq[word] = 1
        else:
            self.word2frq[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, item):
        if item in self.word2idx:
            return self.word2idx[item]
        else:
            return self.word2idx['<unk>']

    def rebuild_by_freq(self, thd=3):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']

        for k, v in self.word2frq.items():
            if v >= thd and (not k in self.idx2word):
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1

        print('Number of words:', len(self.idx2word))
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        dict_file_name = os.path.join(path, 'dict.pkl')
        if False and os.path.exists(dict_file_name):
            self.dictionary = pickle.load(open(dict_file_name, 'rb'))
        else:
            self.dictionary = Dictionary()
            self.add_words(train_file_ids)
            self.dictionary.rebuild_by_freq()
            pickle.dump(self.dictionary, open(dict_file_name, 'wb'))

        self.train, self.train_sens, self.train_trees, self.train_nltktrees = self.tokenize(train_file_ids)
        self.test, self.test_sens, self.test_trees, self.test_nltktrees = self.tokenize(test_file_ids)

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            if tag in word_tags:
                w = w.lower()
                w = re.sub('[0-9]+', 'N', w)
                new_w = w
                words.append(new_w)

        return words

    def filter_words_tag(self, tree):
        words = []
        for w, tag in tree.pos():
            if tag in word_tags:
                w = w.lower()
                new_w = w
                if tag in ["CD", "LS", "JJ", "NNP", "RB"] and re.match(P, w) and not re.match(PY, w)  and not re.match(PB, w) or w == "%":
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

    def add_words(self, file_ids):
        # Add words to the dictionary
  
        for id in file_ids:
            sentences = ptb.parsed_sents(id)
            for sen_tree in sentences:
                words = self.filter_words(sen_tree)
                words = words + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, file_ids):

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
            
        sens_idx = []
        sens = []
        trees = []
        nltk_trees = []
        N = 0
        for id in file_ids:
            sentences = ptb.parsed_sents(id)
            for sen_tree in sentences:
                words = self.filter_words(sen_tree)
                words = self.filter_words_tag(sen_tree)
                words = words + ['<eos>']

                sens.append(words)
                idx = []
                for word in words:
                    idx.append(self.dictionary[word])
                sens_idx.append(idx)
                trees.append(tree2list(sen_tree))
                nltk_trees.append(sen_tree)
                N += len(words)

        return sens_idx, sens, trees, nltk_trees

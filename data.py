import os
import pickle
from collections import defaultdict

basedir = os.path.abspath(os.path.dirname(__file__))
filedir = os.path.join(basedir, 'stanfordSentimentTreebank')

_UNK = '<UNK>'

class Node:
    def __init__(self, word=None, label=None):
        self.label = label # the class (0 - 5)
        self.word = word # the word, may be None
        self.left = None
        self.right = None

        # Neural Network placeholder
        self.h = None
        self.probs = None

    def add_child(self, node):
        if not self.left:
            self.left = node
        elif not self.right:
            self.right = node
        else:
            raise Exception('This tree already has a left and right:', node)

    @property
    def is_leaf(self):
        return not self.left and not self.right

    def get_leaves(self):
        leaves = []
        q = [self]
        while q:
            e = q.pop()

            if e.left:
                q.append(e.left)
            if e.right:
                q.append(e.right)

            if e.is_leaf:
                leaves.append(e)
        return leaves

    def _replace_words_with_nums(self, word_dict):
        if self.is_leaf:
            self.word = word_dict.get(self.word, word_dict[_UNK])
        else:
            self.left._replace_words_with_nums(word_dict)
            self.right._replace_words_with_nums(word_dict)

    def __str__(self):
        if self.is_leaf:
            return '(%d %s)' % (self.label, str(self.word))

        ret = '(' + str(self.label)
        if self.left:
            ret += ' '
            ret += str(self.left)

        if self.right:
            ret += ' '
            ret += str(self.right)

        return ret + ')'

######
# Read datasets in PTB form
######

def build_tree_from_ptb(line):
    line = line.strip()
    assert line[0] == '(', line
    assert line[-1] == ')', line

    label = int(line[1]) - 1
    node = Node(label=int(line[1]) - 1)

    index = 2
    while line[index] == ' ':
        index += 1

    open_parens = 0
    close_parens = 0
    if line[index] == '(':
        start_subtree = index
        open_parens += 1
        index += 1

    while open_parens != close_parens:
        if line[index] == '(':
            open_parens += 1
        elif line[index] == ')':
            close_parens += 1
        index += 1

    if open_parens == 0: # leaf
        node.word = line[index:-1]
    else:
        left_node = build_tree_from_ptb(line[start_subtree:index])
        right_node = build_tree_from_ptb(line[index:-1])
        node.add_child(left_node)
        node.add_child(right_node)

    return node

def get_word_dict(trees, word_in=None, write=True):
    if word_in:
        return pickle.load(open(word_in, 'rb'))

    word_in = os.path.join(filedir, 'word_dict.p')

    words = set([_UNK])
    for root in trees:
        q = [root]
        while q:
            node = q.pop()
            if node.is_leaf:
                words.add(node.word)
            else:
                q.append(node.left)
                q.append(node.right)

    words = dict(zip(words, xrange(len(words))))
    if write:
        pickle.dump(words, open(word_in, 'wb'))
    return words

def read_ptb_dataset(ptb_in, word_in=None):
    with open(ptb_in, 'rb') as f:
        trees = [build_tree_from_ptb(line.strip()) for line in f.readlines()]

    word_dict = get_word_dict(trees, word_in)
    for tree in trees:
        tree._replace_words_with_nums(word_dict)

    return trees, word_dict

######
# Write datasets from original data
######

def _connect(parent_pointers, subtrees, connected, num):
    parent_pointer = parent_pointers[num]

    if connected.get(num) or parent_pointer < 0:
        return

    subtrees[parent_pointer].add_child(subtrees[num])
    connected[num] = True
    _connect(parent_pointers, subtrees, connected, parent_pointer)

def _transform_parens(word):
    if word == '(':
        return '-LRB-'
    elif word == ')':
        return '-RRB-'
    return word

def _make_tree(parent_pointers, sentence, phrases, scores):
    max_node = max([p for p in parent_pointers])

    subtrees = []
    for word in sentence:
        subtrees.append(Node(word=word))

    for _ in xrange(len(sentence), max_node + 1):
        subtrees.append(Node())

    connected = {}
    root = None
    for num, e in enumerate(parent_pointers):
        if e == -1:
            if root is None:
                raise Exception('Found two roots for sentence:', sentence)
            root = subtrees[num]
        else:
            _connect(parent_pointers, subtrees, connected, num)

    for num in xrange(maxNode + 1):
        leaves = subtrees[num].get_leaves()
        words = [leaf.word for leaf in leaves]
        phrase_key = [_transform_parens(word) for word in words]
        if phrase_key in phrases:
            phrase_id = phrases[phrase_key]
        elif words in phrases:
            phrase_id = phrases[words]
        else:
            raise Exception('Could not find phrase id for phrase', sentence)

        score = scores.get(phrase_id)
        if not score:
            raise Exception('Could not find score for phrase id', phrase_id)

        class_label = int(round(math.floor(score * 5)))
        if class_label > 4:
            class_label = 4
        subtrees[num].label = str(class_label)

    # Here there's a shit ton of pattern matching shit performed with TSurgeon
    # Is this all necessary?

    return root

def write_trees(fout, trees, tree_ids):
    with open(fout, 'wb') as f:
        for tree_id in tree_ids:
            tree = trees.get(tree_id)


def write_datasets(train_file=None, test_file=None, dev_file=None):
    train_file = train_file or os.path.join(filedir, 'train.txt')
    test_file = test_file or os.path.join(filedir, 'test.txt')
    dev_file = dev_fie or os.path.join(filedir, 'dev.txt')

    split_path = os.path.join(filedir, 'datasetSplit.txt')
    dictionary_path = os.path.join(filedir, 'dictionary.txt')
    parse_path = os.path.join(filedir, 'STree.txt')
    sentiment_path = os.path.join(filedir, 'sentiment_labels.txt')
    tokens_path = os.path.join(filedir, 'SOStr.txt')

    with open(tokens_path, 'rb') as f:
        sentences = [line.strip().split('|') for line in f.readlines()]

    with open(dictionary_path, 'rb') as f:
        phrases = {}
        for line in f.readlines():
            phrase, id_ = line.split('|')
            phrase = phrase.strip()
            id_ = int(id_.strip())
            phrases[phrase.split()] = id_

    with open(sentiment_path, 'rb') as f:
        scores = {}
        for line in f.readlines():
            if line.startswith('phrase'):
                continue
            id_, score = line.split('|')
            id_ = int(id_.strip())
            score = float(score.strip())
            scores[id_] = score

    with open(parse_path, 'rb') as f:
        trees = []
        for num, line in enumerate(f.readlines()):
            words = line.strip().split('|') # number reps for words
            parent_pointers = [int(word) - 1 for word in words]
            trees.append(
                _make_tree(parent_pointers, sentences[num], phrases, scores)
                )

    with open(split_path, 'rb') as f:
        splits = {1:[], 2:[], 3:[]}
        for line in f.readlines():
            if line.startswith('sentence_index'):
                continue
            tree_id, file_id = line.split(',')
            tree_id = int(tree_id.strip()) - 1
            file_id = int(file_id.strip())
            splits[file_id].append(tree_id)

    trainTree = write_trees(train_file, trees, splits[1])
    testTree = write_trees(test_file, trees, splits[2])
    devTree = write_trees(dev_file, trees, splits[3])

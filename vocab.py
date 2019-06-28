import nltk
import pickle
import argparse
import os
from collections import Counter
import json

PATH_TO_VOCAB = 'vocab.pkl'
### Vocabulary class
class Vocabulary(object):
    """a vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def start_token(self):
        return '<start>'

    def end_token(self):
        return '<end>'


def build_vocab(path='relative_captions_shoes.json', threshold=1, max_words=15000):
    """Build a vocabulary wrapper."""
    with open(path, 'r') as f:
            print("Load json file from {}".format(path))
            data = json.load(f)
            
    caption = [ind['RelativeCaption'] for ind in data]
    counter = Counter()
    for ind in caption:
        tokens = nltk.tokenize.word_tokenize(ind.lower())
        counter.update(tokens)
    # 4 special tokens
    words = counter.most_common(max_words-4)
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in words if cnt >= threshold]

    # Creates a vocabulary wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word(vocab.start_token())
    vocab.add_word(vocab.end_token())
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    print('Total number of words in vocab:', len(words))
    return vocab

def dump_vocab(path=PATH_TO_VOCAB):
    if not os.path.exists(path):
        vocab = build_vocab()
        with open(path, 'wb') as f:
            pickle.dump(vocab, f)
        print("Total vocabulary size: %d" %len(vocab))
        print("Saved the vocabulary wrapper to '%s'" %path)
    else:
        print('Vocabulary already exists.')

def load_vocab(path=PATH_TO_VOCAB):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError('Failed to load %s: %s' % (path, e))


def main():
    vocab = build_vocab(path='relative_captions_shoes.json')
    dump_vocab(PATH_TO_VOCAB)


if __name__ == '__main__':
    main()
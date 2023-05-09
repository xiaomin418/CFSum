import os
import collections
import logging
logger = logging.getLogger(__name__)
VOCAB_NAME = 'vocab.txt'

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

class MyTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, max_size=50000):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    max_size, self._count))
                    break

        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
        self._count, self._id_to_word[self._count - 1]))

        self.SENTENCE_START = '<s>'
        self.SETENCE_END = '</s>'

        self.PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
        self.UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
        self.START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
        self.STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequencesN

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def __len__(self):
        return self._count

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def tokenize(self, text):
        text = text.split()
        split_tokens = [self.word2id(w) for w in text]
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self._word_to_id[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self._id_to_word[i])
        return tokens



import os
import re
import string
import gensim.downloader
from collections import Counter
import numpy as np
import scipy.sparse
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from utils import file_utils


# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')

def get_stopwords(stopwords):
    if isinstance(stopwords, list):
        stopword_set = stopwords
    elif isinstance(stopwords, str):
        stopword_set = file_utils.read_texts(stopwords)
    else:
        raise NotImplementedError(stopwords)

    return stopword_set

class Tokenizer:
    def __init__(self, stopwords, keep_num, keep_alphanum, strip_html, no_lower, min_length):
        self.keep_num = keep_num
        self.keep_alphanum = keep_alphanum
        self.strip_html = strip_html
        self.lower = not no_lower
        self.min_length = min_length

        self.stopword_set = []

    def clean_text(self, text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False):
        # remove html tags
        if strip_html:
            text = re.sub(r'<[^>]+>', '', text)
        else:
            # replace angle brackets
            text = re.sub(r'<', '(', text)
            text = re.sub(r'>', ')', text)
        # lower case
        if lower:
            text = text.lower()
        # eliminate email addresses
        if not keep_emails:
            text = re.sub(r'\S+@\S+', ' ', text)
        # eliminate @mentions
        if not keep_at_mentions:
            text = re.sub(r'\s@\S+', ' ', text)
        # replace underscores with spaces
        text = re.sub(r'_', ' ', text)
        # break off single quotes at the ends of words
        text = re.sub(r'\s\'', ' ', text)
        text = re.sub(r'\'\s', ' ', text)
        # remove periods
        text = re.sub(r'\.', '', text)
        # replace all other punctuation (except single quotes) with spaces
        text = replace.sub(' ', text)
        # remove single quotes
        text = re.sub(r'\'', '', text)
        # replace all whitespace with a single space
        text = re.sub(r'\s', ' ', text)
        # strip off spaces on either end
        text = text.strip()
        return text

    def tokenize(self, text, vocab=None):
        #text = self.clean_text(text, self.strip_html, self.lower)
        tokens = text

        tokens = ['_' if t in self.stopword_set else t for t in tokens]

        # drop short tokens
        """
        if self.min_length > 0:
            tokens = [t if len(t) >= self.min_length else '_' for t in tokens]
        """
        #print('token',tokens)
        unigrams = [t for t in tokens if t != '_']
        # counts = Counter()
        # counts.update(unigrams)
        #print('uni',unigrams)

        tokens = [token for token in unigrams]
     
        
        return tokens
class Preprocessing:
    def __init__(self, test_sample_size=None, test_p=0.2, stopwords="snowball", min_doc_count=3, max_doc_freq=1.0, keep_num=False, keep_alphanum=False, strip_html=False, no_lower=False, min_length=3, min_term=1, vocab_size=5000, seed=42):
        """
        Args:
            test_sample_size:
                Size of the test set.
            test_p:
                Proportion of the test set. This helps sample the train set based on the size of the test set.
            stopwords:
                List of stopwords to exclude [None|mallet|snowball].
            min-doc-count:
                Exclude words that occur in less than this number of documents.
            max_doc_freq:
                Exclude words that occur in more than this proportion of documents.
            keep-num:
                Keep tokens made of only numbers.
            keep-alphanum:
                Keep tokens made of a mixture of letters and numbers.
            strip_html:
                Strip HTML tags.
            no-lower:
                Do not lowercase text
            min_length:
                Minimum token length.
            min_term:
                Minimum term number
            vocab-size:
                Size of the vocabulary (by most common in the union of train and test sets, following above exclusions)
            seed:
                Random integer seed (only relevant for choosing test set)
        """

        self.test_sample_size = test_sample_size
        self.min_doc_count = min_doc_count
        self.max_doc_freq = max_doc_freq
        self.min_term = min_term
        self.test_p = test_p
        self.vocab= []
        self.seed = seed
        self.vocab_size=5000
        self.tokenizer = Tokenizer(stopwords, keep_num, keep_alphanum, strip_html, no_lower, min_length)

    def parse(self, texts, vocab):
        if not isinstance(texts, list):
            texts = [texts]

        parsed_texts = list()
        for i, text in enumerate(tqdm(texts, desc="===>parse texts")):
            tokens = tokens = self.tokenizer.tokenize(text, vocab=vocab)
            parsed_texts.append(' '.join(tokens))

        vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())
        bow_matrix = vectorizer.fit_transform(parsed_texts)
        bow_matrix = bow_matrix.toarray()
        return parsed_texts, bow_matrix

    def parse_dataset(self, dataset_train):

        #print('dataset_train',dataset_train)
        word_counts = Counter()
        doc_counts_counter = Counter()

        tokens = self.tokenizer.tokenize(dataset_train)
        #print(tokens)
        word_counts.update(tokens)
        doc_counts_counter.update(set(tokens))
        parsed_text = ' '.join(tokens)
        # train_texts and test_texts have been parsed.
        words, doc_counts = zip(*doc_counts_counter.most_common())
        n_items=len(dataset_train)
        doc_freqs = np.array(doc_counts) / float(n_items)
        vocab = [word for i, word in enumerate(words) if doc_counts[i] >= self.min_doc_count and doc_freqs[i] <= self.max_doc_freq]
        print('len',len(vocab))
        # filter vocabulary
        #if (self.vocab_size is not None) and (len(vocab) > self.vocab_size):
        vocab = vocab[:self.vocab_size]

        vocab.sort()
        return vocab
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
from nltk.corpus import stopwords as stop_words
import warnings
import numpy as np

# ----- Original -----
class WhiteSpacePreprocessing():
    """
    Provides a very simple preprocessing script that filters infrequent tokens from text
    """
    def __init__(self, documents, stopwords_language="english", vocabulary_size=2000, min_len=10, max_len=200):
        """

        :param documents: list of strings
        :param stopwords_language: string of the language of the stopwords (see nltk stopwords)
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.stopwords = set(stop_words.words(stopwords_language) + stop_words.words("english"))
        self.vocabulary_size = vocabulary_size
        self.max_len = max_len
        self.min_len = min_len

    def preprocess(self):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
        print("Preprocessing", len(self.documents), "documents")
        print("Max seq length:", self.max_len)

        # --my changes-- truncate raw articles to the first 200 tokens
        truncated_docs = [' '.join(doc.split()[:self.max_len]) for doc in self.documents]
        self.documents = truncated_docs
        # --end my changes --
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 2 and w not in self.stopwords])
                             for doc in preprocessed_docs_tmp]

        vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
        vectorizer.fit_transform(preprocessed_docs_tmp)
        vocabulary = set(vectorizer.get_feature_names())
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary])
                                 for doc in preprocessed_docs_tmp]

        preprocessed_docs, unpreprocessed_docs = [], []
        for i, doc in enumerate(preprocessed_docs_tmp):
            if len(doc.split()) >= self.min_len:
                # if len(doc.split()) > self.max_len:
                #     doc = ' '.join(doc.split()[:self.max_len])
                # if len(self.documents[i].split()) > self.max_len:
                #     self.documents[i] = ' '.join(self.documents[i].split()[:self.max_len])
                preprocessed_docs.append(doc)
                unpreprocessed_docs.append(self.documents[i])

        return preprocessed_docs, unpreprocessed_docs, list(vocabulary)


# ----- Multimodal AND Multilingual (M3L) -----

class WhiteSpacePreprocessingM3L():
    """
    Provides a very simple preprocessing script for aligned multimodal AND multilingual data
    """
    def __init__(self, documents, image_urls, stopwords_languages, vocabulary_size=4000, min_len=10, max_len=200):
        """

        :param documents: list of lists of strings, e.g. [['good morning', 'thank you'], ['guten morgen', 'danke sie']]
        :param stopwords_language: list  of strings of the languages (see nltk stopwords)
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.image_urls = image_urls
        self.num_lang = len(stopwords_languages)
        self.languages = stopwords_languages
        self.stopwords = []
        for lang in self.languages:
            self.stopwords.append(set(stop_words.words(lang)))
        # same vocab_size for all langs for now
        self.vocabulary_size = vocabulary_size
        self.min_len = min_len
        self.max_len = max_len
        # if user has custom stopwords list
        #self.custom_stops = custom_stops

    def preprocess(self):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.

        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
      
        for l in range(self.num_lang):
            truncated_docs = [' '.join(doc.split()[:self.max_len]) for doc in self.documents[l]]
            self.documents[l] = truncated_docs

        preprocessed_docs_tmp = []
        vocabulary = []
        for l in range(self.num_lang):
            preprocessed_docs = [doc.lower() for doc in self.documents[l]]
            preprocessed_docs = [doc.translate(
                str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs]
            preprocessed_docs = [' '.join([w for w in doc.split() if len(w) > 2 and w not in self.stopwords[l]])
                                 for doc in preprocessed_docs]
            vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
            vectorizer.fit_transform(preprocessed_docs)
            vocabulary_lang = set(vectorizer.get_feature_names())
            preprocessed_docs = [' '.join([w for w in doc.split() if w in vocabulary_lang]) for doc in preprocessed_docs]
            preprocessed_docs_tmp.append(preprocessed_docs)
            vocabulary.append(list(vocabulary_lang))
            # print('vocab size:', len(vocabulary_lang))

        if self.num_lang == 1:
            preprocessed_data_final = [[]]
            unpreprocessed_data_final = [[]]
            image_urls_final = []

            for i in range(len(preprocessed_docs_tmp[0])):
                doc1 = preprocessed_docs_tmp[0][i]
                image_url = self.image_urls[i]
                if self.min_len <= len(doc1.split()):
                    preprocessed_data_final[0].append(doc1)

                    unpreprocessed_data_final[0].append(self.documents[0][i])

                    image_urls_final.append(image_url)


        elif self.num_lang == 2:

            preprocessed_data_final = [[], []]
            unpreprocessed_data_final = [[], []]
            image_urls_final = []
            # docs must be aligned across languages and modalities (text-image)
            for i in range(len(preprocessed_docs_tmp[0])):
                doc1 = preprocessed_docs_tmp[0][i]
                doc2 = preprocessed_docs_tmp[1][i]
                image_url = self.image_urls[i]
                if self.min_len <= len(doc1.split()) and self.min_len <= len(doc2.split()):
                    preprocessed_data_final[0].append(doc1)
                    preprocessed_data_final[1].append(doc2)

                    unpreprocessed_data_final[0].append(self.documents[0][i])
                    unpreprocessed_data_final[1].append(self.documents[1][i])

                    image_urls_final.append(image_url)

        else:
            # TODO: rewrite in generic form for any number of languages
            raise NonImplementedError("Cannot process number of languages: %s" %self.num_lang)


        # preprocessed_data_final is a list of list of strings (processed articles) and image urls
        # unpreprocessed_data_final is a list of list of strings (original articles) and image urls
        # vocabulary is a list of list of words (separate vocabularies for each language)
        return preprocessed_data_final, unpreprocessed_data_final, vocabulary, image_urls_final


# ----- Multilingual only -----
class WhiteSpacePreprocessingMultilingual():
    """
    Provides a very simple preprocessing script for aligned multilingual documents
    """
    def __init__(self, documents, stopwords_languages, vocabulary_size=20000, min_len=10, custom_stops=None, max_len=4000):
        """

        :param documents: list of lists of strings, e.g. [['good morning', 'thank you'], ['guten morgen', 'danke sie']]
        :param stopwords_language: list  of strings of the languages (see nltk stopwords)
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.num_lang = 2
        self.languages = stopwords_languages
        self.stopwords = []
        for lang in self.languages:
            self.stopwords.append(set(stop_words.words(lang)))
        # same vocab_size for all langs for now
        self.vocabulary_size = vocabulary_size
        # min/max article length after preprocessing
        self.min_len = min_len
        self.max_len = max_len
        # if user has custom stopwords list
        self.custom_stops = custom_stops
        #self.num_trainer=125
     
       
    def preprocess(self):
       
        """
        preprocessed_docs_tmp = []
        for l in range(self.num_lang):
            print("--- lang", l, ":", self.languages[l], "---")
            preprocessed_docs = [doc.lower() for doc in self.documents[l]]
            preprocessed_docs = [doc.translate(
                str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs]
            preprocessed_docs = [' '.join([w for w in doc.split() if len(w) > 2 and w not in self.stopwords[l]])
                                 for doc in preprocessed_docs]
            if self.custom_stops is not None:
                preprocessed_docs = [' '.join([w for w in doc.split() if len(w) > 2 and w not in self.custom_stops[l]])
                                     for doc in preprocessed_docs]
            vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
            vectorizer.fit_transform(preprocessed_docs)
            vocabulary_lang = set(vectorizer.get_feature_names())
            preprocessed_docs = [' '.join([w for w in doc.split() if w in vocabulary_lang]) for doc in preprocessed_docs]
            preprocessed_docs_tmp.append(preprocessed_docs)
            
        preprocessed_docs_final = [[], []]
        unpreprocessed_docs_final = [[], []]
        # docs must be aligned across languages
        for i in range(len(preprocessed_docs_tmp[0])):
            doc1 = preprocessed_docs_tmp[0][i]
            doc2 = preprocessed_docs_tmp[1][i]
            if self.min_len <= len(doc1.split()) and self.min_len <= len(doc2.split()):
                # truncate docs if they exceed max_len
                # if len(doc1.split()) > self.max_len:
                #     doc1 = " ".join(doc1.split()[:self.max_len])
                # if len(doc2.split()) > self.max_len:
                #     doc2 = " ".join(doc2.split()[:self.max_len])
                preprocessed_docs_final[0].append(doc1)
                preprocessed_docs_final[1].append(doc2)
                unpreprocessed_docs_final[0].append(self.documents[0][i])
                unpreprocessed_docs_final[1].append(self.documents[1][i])
        
      
        # truncate raw articles to the first max_len tokens
        """
        for l in range(self.num_lang):
           for doc in self.documents[l]:
             if isinstance(doc, str):
               truncated_docs = [' '.join(doc.split()[:self.max_len])]
           self.documents[l].append(truncated_docs)
        preprocessed_docs_tmp=[[] for i in range(2)]
        preprocessed_docs= [[] for i in range(2)]
        vocabulary = []
        
        for j in range (min(20000,len(self.documents[0]))):
            preprocessed_docs[0].append(str(self.documents[0][j]))
            preprocessed_docs[1].append(str(self.documents[1][j]))
            #print('p1',preprocessed_docs[1])
        preprocessed_docs[0] = [doc.translate(
           str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs[0]]
        preprocessed_docs[0] = [' '.join([w for w in doc.split() if len(w) > 2 ])
                                 for doc in preprocessed_docs[0]]
            #print('p0',preprocessed_docs[0])
        preprocessed_docs[1] = [doc.translate(
               str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs[1]]
        preprocessed_docs[1] = [' '.join([w for w in doc.split() if len(w) > 2 ])
                                 for doc in preprocessed_docs[1]]
            #print('p1',preprocessed_docs[1])
        vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
        vectorizer.fit_transform(preprocessed_docs[0])
        vocabulary_lang0 = set(vectorizer.get_feature_names_out())
        vectorizer.fit_transform(preprocessed_docs[1])
        vocabulary_lang1 = set(vectorizer.get_feature_names_out())
        #print('vocabulary_lang0:',vocabulary_lang0)
        #print('vocabulary_lang1:',vocabulary_lang1)
        vocabulary_lang = vocabulary_lang0.union(vocabulary_lang1)
        preprocessed_docs0 = [' '.join([w for w in doc.split() if w in vocabulary_lang0]) for doc in preprocessed_docs[0]]
        preprocessed_docs1 = [' '.join([w for w in doc.split() if w in vocabulary_lang1]) for doc in preprocessed_docs[1]]
        preprocessed_docs_tmp[0].append(preprocessed_docs0)
        preprocessed_docs_tmp[1].append(preprocessed_docs1)
        vocabulary.append(list(vocabulary_lang0))
        vocabulary.append(list(vocabulary_lang1))
        """
       
        for i in range(len(preprocessed_docs0)):
           preprocessed_docs_tmp[0].append(preprocessed_docs0[i])
           preprocessed_docs_tmp[1].append(preprocessed_docs1[i])
        vocabulary[0].append(list(vocabulary_lang0))
        vocabulary[1].append(list(vocabulary_lang1))
            
        #print('t0:', len(preprocessed_docs0))
        """
        preprocessed_docs_final = [[], []]
        unpreprocessed_docs_final = [[], []]
        #print(len(self.documents[0]))
        #print(len(preprocessed_docs0))
        for i in range(min(20000,len(self.documents[0]))):
            preprocessed_docs_final[0].append(preprocessed_docs0[i])
            preprocessed_docs_final[1].append(preprocessed_docs1[i])
            unpreprocessed_docs_final[0].append(self.documents[0][i])
            unpreprocessed_docs_final[1].append(self.documents[1][i])
        #print(preprocessed_docs_final[1])
        # preprocessed_docs_final is a list of list of strings (processed articles)
        # unpreprocessed_docs_final is a list of list of strings (original articles)
        # vocabulary is a list of list of words (separate vocabularies for each language)
        print('p',len(preprocessed_docs_final[0]))
        print('un',len(unpreprocessed_docs_final[0]))
        return preprocessed_docs_final, unpreprocessed_docs_final, vocabulary
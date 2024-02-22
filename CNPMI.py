import os
import math
import threading
from multiprocessing import Pool
import argparse
from sklearn.feature_extraction.text import CountVectorizer

from utils import file_utils


wc_lock = threading.Lock()
global_word_count = dict()  # global. both single and pair.
WTOTALKEY = "!!<TOTAL_WINDOWS>!!"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics1',default='/home/ssliang/M3L-topic-model/output/ECNews_en_7--topic=40.txt')
    parser.add_argument('--topics2',default='/home/ssliang/M3L-topic-model/output/ECNews_cn_7--topic=40.txt')
    parser.add_argument('--ref_corpus_config', type=str,default="/home/ssliang/M3L-topic-model/configs/ref_corpus/en_zh.yaml")
    parser.add_argument('--metric', type=str, default='npmi')
    args = parser.parse_args()
    return args


def calcwcngram_complete(worker_wordcount):
    wc_lock.acquire()

    #update the global wordcount from the worker
    for k, v in worker_wordcount.items():
        curr_v = 0
        if k in global_word_count:
            curr_v = global_word_count[k]
        curr_v += v
        global_word_count[k] = curr_v
    #print('global_word_count',global_word_count)
    wc_lock.release()
def read_texts(path):
    texts = list()
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            texts.append(line.strip(' '))
    return texts

def make_bow(corpus_path, vocab):
    corpus = file_utils.read_texts(corpus_path)
    # NOTE: use x.split(). single character words will NOT be removed.
    # NOTE: the output vocabulary order will be the same as vocab.
    vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())
    bow = vectorizer.transform(corpus).toarray()
    wc_dict = dict(zip(vocab, bow.sum(0)))
    return bow, wc_dict


def calcwcngram(corpus_tuple, vocab1, vocab2, word_pair_list, sep_token):

    print("===>calculating word counts.")

    corpus1_path, corpus2_path = corpus_tuple
    all_wc_dict = dict()
    bow1, wc_dict1 = make_bow(corpus1_path, vocab1)
    bow2, wc_dict2 = make_bow(corpus2_path, vocab2)

    all_wc_dict.update(wc_dict1)
    all_wc_dict.update(wc_dict2)

    for word_pair in word_pair_list:
        if word_pair not in all_wc_dict:
            w1, w2 = word_pair.split(sep_token)
            pair_count = ((bow1[:, vocab1.index(w1)] * bow2[:, vocab2.index(w2)]) > 0).sum()
            all_wc_dict[word_pair] = pair_count
    
    # the number of all windows. Here each doc is one window.
    all_wc_dict[WTOTALKEY] = bow1.shape[0]
   
    return all_wc_dict


#compute the association between two words
def calc_assoc(word_pair, window_total, sep_token, metric):
    word1, word2 = word_pair.split(sep_token)

    combined_count = float(global_word_count[word_pair])
    w1_count = float(global_word_count[word1])
    w2_count = float(global_word_count[word2])

    if (metric == "pmi") or (metric == "npmi"):
        if w1_count == 0 or w2_count == 0 or combined_count == 0:
            result = 0.0
        else:
            result = math.log((combined_count * window_total) / (w1_count * w2_count), 10)
            if metric == "npmi":
                result = result / (-1.0 * math.log(combined_count / window_total, 10))

    elif metric == "lcp":
        if combined_count == 0:
            if w2_count != 0:
                result = math.log(w2_count / window_total, 10)
            else:
                result = math.log(1.0 / window_total, 10)
        else:
            result = math.log(combined_count / w1_count, 10)
    #print('result',result)
    return result


def main():
    args = parse_args()

    parallel_corpus_tuples = file_utils.read_yaml(args.ref_corpus_config)['parallel_corpus_tuple']
    sep_token = '|'
    topic1 = read_texts(args.topics1)
    topics1 = []
    for i in range(len(topic1)):
       topics1.append(topic1[i].strip('\n').split(' '))
    #print(topics1)
    topic2 = read_texts(args.topics2)
    topics2 = []
    for i in range(len(topic2)):
       topics2.append(topic2[i].strip('\n').split(' '))
    num_topic = len(topics1)
    num_top_word = len(topics1[0])

    print("===>preparing word pairs.")
    vocab1 = set([])
    vocab2 = set([])
    word_pair_list = list()
    for k in range(num_topic):
        for i in range(num_top_word):
            w1 = topics1[k][i]
            vocab1.add(w1)
            for j in range(num_top_word):
                w2 = topics2[k][j]
                vocab2.add(w2)
                word_pair_list.append(f'{w1}{sep_token}{w2}')

    word_pair_list = tuple(word_pair_list)
    #print(word_pair_list)
    vocab1 = sorted(list(vocab1))
    vocab2 = sorted(list(vocab2))

    pool = Pool()
    #print(parallel_corpus_tuples)
    for i, cp in enumerate(parallel_corpus_tuples):
        #print(cp[0],' ',cp[1])
        if not os.path.exists(cp[0]):
            raise FileNotFoundError(cp[0])
        if not os.path.exists(cp[1]):
            raise FileNotFoundError(cp[1])

        print("===>creating a thread for parallel corpus: ", cp)
        param_list = (cp, vocab1, vocab2, word_pair_list, sep_token)
        pool.apply_async(calcwcngram, param_list, callback=calcwcngram_complete)

    # wait for the subprocesses.
    pool.close()
    pool.join()
    #print('===>global_word_count',global_word_count)
    print("===>computing coherence metric.")
    topic_assoc = list()
    
    window_total = float(global_word_count[WTOTALKEY])
    for word_pair in word_pair_list:
        topic_assoc.append(calc_assoc(word_pair, window_total, sep_token, metric=args.metric))

    result = float(sum(topic_assoc)) / len(topic_assoc)
    print(f'{result:.5f}')


if __name__ == '__main__':
    main()

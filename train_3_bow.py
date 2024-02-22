import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import torch
import string
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from models.trilingual_contrast_bow import TrilingualContrastiveTM
from utils.data_preparation import TrilingualTopicModelDataPreparation
from utils.preprocessing import WhiteSpacePreprocessingMultilingual
from utils import file_utils
import wordcloud
import scipy
from collections import defaultdict
ignore_mismatched_sizes=True
import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from process_vocab import Preprocessing
from utils import file_utils
import matplotlib.pyplot as plt
import pickle


argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', default='MultilingualContrast', type=str)
argparser.add_argument('--data_path1', default='/home/ssliang/M3L-topic-model/data/', type=str)
argparser.add_argument('--data_path2', default='/home/ssliang/M3L-topic-model/data2/Rakuten_Amazon/', type=str)
argparser.add_argument('--save_path', default='trained_models/', type=str)
argparser.add_argument('--num_topics', default=40, type=int)
argparser.add_argument('--num_epochs', default=150, type=int)
argparser.add_argument('--langs', default='en,cn,ja', type=str)
argparser.add_argument('--sbert_model', default='/home/ssliang/M3L-topic-model/multilingual-mpnet', type=str)
argparser.add_argument('--text_enc_dim', default=768, type=int)
argparser.add_argument('--batch_size', default=1024, type=int)
argparser.add_argument('--max_seq_length', default=20000, type=int)
argparser.add_argument('--kl_weight', default=0.01, type=int, help='weight for the KLD loss')
argparser.add_argument('--cl_weight', default=50, type=int, help='weight for the contrastive loss')
argparser.add_argument('--use_pretrained', default=False, type=bool, help='if use_pretrained or not')

args = argparser.parse_args()
def parse_dictionary(dict_path,word2id_cn,word2id_en,vocab_size_en=20000,vocab_size_cn=20000):
        trans_dict = defaultdict(set)

        trans_matrix_en = np.zeros((vocab_size_en,vocab_size_cn), dtype='int32')
        trans_matrix_cn = np.zeros((vocab_size_cn,vocab_size_en), dtype='int32')

        dict_texts = file_utils.read_texts(dict_path)

        for line in dict_texts:
            terms = (line.strip()).split()
            if len(terms) == 2:
                cn_term = terms[0]
                en_term = terms[1]
                if cn_term in word2id_cn and en_term in word2id_en:
                    trans_dict[cn_term].add(en_term)
                    trans_dict[en_term].add(cn_term)
                    cn_term_id = word2id_cn[cn_term]
                    en_term_id = word2id_en[en_term]

                    trans_matrix_en[en_term_id][cn_term_id] = 1
                    trans_matrix_cn[cn_term_id][en_term_id] = 1

        return trans_dict, trans_matrix_en, trans_matrix_cn

print("-"*50 + "\n")
data_dir='/home/ssliang/M3L-topic-model/data2'
train_texts_en1 = file_utils.read_texts(os.path.join(data_dir, 'train_texts_en.txt'))
test_texts_en1 = file_utils.read_texts(os.path.join(data_dir, 'test_texts_en.txt'))
train_texts_cn = file_utils.read_texts(os.path.join(data_dir, 'train_texts_cn.txt'))
test_texts_cn = file_utils.read_texts(os.path.join(data_dir, 'test_texts_cn.txt'))
pretrain_word_embeddings_en1 = scipy.sparse.load_npz(os.path.join(data_dir, f'word2vec_en.npz')).toarray()
pretrain_word_embeddings_cn = scipy.sparse.load_npz(os.path.join(data_dir, f'word2vec_cn.npz')).toarray()

print("-"*50 + "\n")
data_dir='/home/ssliang/M3L-topic-model/data2'
train_texts_en2 = file_utils.read_texts(os.path.join(data_dir, 'train_texts_en1.txt'))
test_texts_en2 = file_utils.read_texts(os.path.join(data_dir, 'test_texts_en1.txt'))
train_texts_ja = file_utils.read_texts(os.path.join(data_dir, 'train_texts_ja.txt'))
test_texts_ja = file_utils.read_texts(os.path.join(data_dir, 'test_texts_ja.txt'))
pretrain_word_embeddings_en2 = scipy.sparse.load_npz(os.path.join(data_dir, f'word2vec_en.npz')).toarray()
pretrain_word_embeddings_ja = scipy.sparse.load_npz(os.path.join(data_dir, f'word2vec_ja.npz')).toarray()


vocab_en1=pickle.load(open('/home/ssliang/M3L-topic-model/data2/vocab_en.pkl','rb'))
vocab_cn=pickle.load(open('/home/ssliang/M3L-topic-model/data2/vocab_cn.pkl','rb'))
vocab_en2=pickle.load(open('/home/ssliang/M3L-topic-model/data2/vocab_en1.pkl','rb'))
vocab_ja=pickle.load(open('/home/ssliang/M3L-topic-model/data2/vocab_ja.pkl','rb'))
vocab_en = file_utils.read_texts(os.path.join(data_dir, 'vocab_en.txt'))
word2id_en1 = dict(zip(vocab_en, range(len(vocab_en))))
word2id_cn = dict(zip(vocab_en, range(len(vocab_en))))
word2id_en2 = dict(zip(vocab_en, range(len(vocab_en))))
word2id_ja = dict(zip(vocab_en, range(len(vocab_en))))
trans_dict1, trans_matrix_en1, trans_matrix_cn = parse_dictionary("/home/ssliang/M3L-topic-model/data/ch_en_dict.dat",word2id_cn,word2id_en1)
trans_dict2, trans_matrix_en2, trans_matrix_ja = parse_dictionary("/home/ssliang/M3L-topic-model/data/ja_en_dict.dat",word2id_ja,word2id_en2)
trans_dict3, trans_matrix_ja2, trans_matrix_cn2= parse_dictionary("/home/ssliang/M3L-topic-model/data/ja_zh_dict.dat",word2id_cn,word2id_ja)
#print(trans_dict)
# stopwords lang dict
vocab1=[vocab_en1,vocab_cn]
vocab2=[vocab_en2,vocab_ja]
raw_docs_train1=[train_texts_en1,train_texts_cn]
raw_docs_train2=[train_texts_en2,train_texts_ja]
#print(vocab_en1)
#print(len(vocab1[0]))
qt = TrilingualTopicModelDataPreparation(args.sbert_model,vocab1,vocab2)
training_dataset = qt.fit_bow(args, text_for_bow1=raw_docs_train1,text_for_bow2=raw_docs_train2)
lang_dict = {'en': 'english',
             'ja': 'Japanese',
             'zh': 'Chinese'}
# ----- load dataset -----

languages = args.langs.lower().split(',')
languages = [l.strip() for l in languages]
#  ----- encode documents -----
raw_docs_text1=[test_texts_en1,test_texts_cn]
raw_docs_text2=[test_texts_en2,test_texts_ja]
qt = TrilingualTopicModelDataPreparation(args.sbert_model,vocab1,vocab2)
validation_dataset = qt.fit_bow(args, text_for_bow1=raw_docs_text1,text_for_bow2=raw_docs_text2)
# ----- initialize model -----
loss_weights = {"KL": args.kl_weight,
                "CL": args.cl_weight}
#print('input_size',qt.vocab_sizes[0])
contrast_model = TrilingualContrastiveTM(
                                           bow_size=args.max_seq_length,
                                           contextual_size=args.text_enc_dim,
                                           n_components=20,
                                           model_type='prodLDA',
                                           num_epochs=args.num_epochs,
                                           languages=languages,
                                           batch_size=args.batch_size,
                                           loss_weights=loss_weights,
                                           pretrain_en1=pretrain_word_embeddings_en1,
                                           pretrain_cn=pretrain_word_embeddings_cn,
                                           pretrain_en2=pretrain_word_embeddings_en2,
                                           pretrain_ja=pretrain_word_embeddings_ja,
                                           vocab_en1=vocab_en1,
                                           vocab_cn=vocab_cn,
                                           vocab_en2=vocab_en2,
                                           vocab_ja=vocab_ja,
                                          )


# ----- topic inference -----
theta1,theta2_new,theta2_1,theta3_1new,beta=contrast_model.fit(training_dataset,validation_dataset)

beta=beta.cpu().detach().numpy()

np.savez("/home/ssliang/M3L-topic-model/data2/theta1--topic=40.npz",theta1.cpu().detach().numpy())
np.savez("/home/ssliang/M3L-topic-model/data2/theta2_new--topic=40.npz", theta2_new.cpu().detach().numpy())
np.savez("/home/ssliang/M3L-topic-model/data2/theta2_1--topic=40.npz", theta2_1.cpu().detach().numpy())
np.savez("/home/ssliang/M3L-topic-model/data2/theta3_1new--topic=40.npz", theta3_1new.cpu().detach().numpy())
np.savez("/home/ssliang/M3L-topic-model/data2/beta.npz",beta)


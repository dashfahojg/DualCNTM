import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import torch
import string
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from models.multilingual_contrast import MultilingualContrastiveTM
from utils.data_preparation import MultilingualTopicModelDataPreparation
from utils.preprocessing import WhiteSpacePreprocessingMultilingual
from measures import CoherenceNPMI,CrosslingualRetrieval,TopicDiversity,JSDivergence
from models.multilingual_contrast import MultilingualContrastiveTM
from utils import file_utils

#print(topics_en)
#model3=CoherenceNPMI(topics_en,text_en).score()
#model4=CoherenceNPMI(topics_cn,text_cn).score()
test_theta1 = np.load("/home/ssliang/M3L-topic-model/data2/theta1--topic=60.npz")["arr_0"]
test_theta2 = np.load("/home/ssliang/M3L-topic-model/data2/theta2_new--topic=60.npz")["arr_0"]
test_theta3 = np.load("/home/ssliang/M3L-topic-model/data2/theta2_1--topic=60.npz")["arr_0"]
test_theta4 = np.load("/home/ssliang/M3L-topic-model/data2/theta3_1new--topic=60.npz")["arr_0"]
print(test_theta1)
JSD1=JSDivergence(test_theta1,test_theta2).score()
JSD2=JSDivergence(test_theta3,test_theta4).score()
print('JSD1',JSD1)
print('JSD2',JSD2)
CR1=CrosslingualRetrieval(test_theta1,test_theta2).mrr_score()
CR2=CrosslingualRetrieval(test_theta3,test_theta4).mrr_score()
print('CR1',CR1)
print('CR2',CR2)
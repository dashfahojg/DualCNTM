import datetime
import multiprocessing as mp
import os
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import wordcloud
import pandas as pd
from scipy.special import softmax
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from utils.earlystopping import EarlyStopping
from utils import file_utils
# decooder network
from networks.decoding_ablation import TrilingualDecoderNetwork
# for contrastive loss
from pytorch_metric_learning.losses import NTXentLoss
import pdb


datas = np.load("/home/ssliang/M3L-topic-model/data2/en_cn.npz")
M_encn=datas['arr_0']
M_encn=torch.tensor(M_encn).cuda()
datas = np.load("/home/ssliang/M3L-topic-model/data2/en_ja.npz")
M_enja=datas['arr_0']
M_enja=torch.tensor(M_enja).cuda()
class TrilingualContrastiveTM:
    """Class to train the contextualized topic model. This is the more general class that we are keeping to
    avoid braking code, users should use the two subclasses ZeroShotTM and CombinedTm to do topic modeling.

    :param bow_size: int, dimension of input
    :param contextual_size: int, dimension of input that comes from BERT embeddings
    :param inference_type: string, you can choose between the contextual model and the combined model
    :param n_components: int, number of topic components, (default 10)
    :param model_type: string, 'prodLDA' or 'LDA' (default 'prodLDA')
    :param hidden_sizes: tuple, length = n_layers, (default (100, 100))
    :param activation: string, 'softplus', 'relu', (default 'softplus')
    :param dropout: float, dropout to use (default 0.2)
    :param learn_priors: bool, make priors a learnable parameter (default True)
    :param batch_size: int, size of batch to use for training (default 64)
    :param lr: float, learning rate to use for training (default 2e-3)
    :param momentum: float, momentum to use for training (default 0.99)
    :param solver: string, optimizer 'adam' or 'sgd' (default 'adam')
    :param num_epochs: int, number of epochs to train for, (default 100)
    :param reduce_on_plateau: bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
    :param num_data_loader_workers: int, number of data loader workers (default cpu_count). set it to 0 if you are using Windows
    :param label_size: int, number of total labels (default: 0)
    :param loss_weights: dict, it contains the name of the weight parameter (key) and the weight (value) for each loss.
    It supports only the weight parameter beta for now. If None, then the weights are set to 1 (default: None).

    """
    
    def __init__(self,bow_size, contextual_size, pretrain_en1,pretrain_cn,pretrain_en2,pretrain_ja,vocab_en1,vocab_cn,
                 vocab_en2,vocab_ja,n_components=40, model_type='prodLDA',hidden_sizes=(100, 100), activation='softplus', dropout=0, 
                 learn_priors=True, batch_size=500,lr=1e-3, momentum=0.99, solver='adam', num_epochs=100, reduce_on_plateau=False,
                 num_data_loader_workers=mp.cpu_count(), label_size=0, loss_weights=None, languages=None):

        self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # langs is an array of language codes e.g. ['en', 'de']
        self.num_lang = 2
        self.bow_size = bow_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        #dropout is same for all langs
        self.dropout = dropout
        self.vocab_en1=vocab_en1
        self.vocab_cn=vocab_cn
        self.vocab_en2=vocab_en2
        self.vocab_ja=vocab_ja
        #learn_priors is same for all langs
        self.learn_priors = learn_priors
        #batch_size is same for all langs
        self.batch_size = batch_size
        #lr is same for all langs
        self.lr = lr
        #contextual_size is same for all langs
        self.contextual_size = contextual_size
        # same
        self.momentum = momentum
        # same
        self.solver = solver
        # same
        self.num_epochs = num_epochs
        # name
        self.num_data_loader_workers = num_data_loader_workers
        self.infnet='zeroshot'
        self.pretrain_en1=pretrain_en1
        self.pretrain_cn=pretrain_cn
        #print(self.pretrain_en)
        # same for now
        #self.BCEloss=F.binary_cross_entropy()
        if loss_weights:
            self.weights = loss_weights
        else:
            self.weights = {"KL": 0.01, "CL": 90}
        # contrastive decoder
        self.model = TrilingualDecoderNetwork(
            bow_size, self.contextual_size, self.infnet, self.n_components, model_type, hidden_sizes, activation,
            dropout, learn_priors)
            
        
        self.early_stopping = None

        # init optimizers
        if self.solver == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum)
        # performance attributes
        self.best_loss_train = float('inf')

        # training attributes
        self.model_dir = None
        self.train_data = None
        self.nn_epoch = None

        # validation attributes
        self.validation_data = None

        # learned topics
        # best_components: n_components x vocab_size
        self.best_components = None

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

        self.model = self.model.to(self.device)
        self.MSEloss = nn.MSELoss()
        
    def loss_info(self,x1,x2, temperature):
        assert len(x1.size()) == 2

        # Cosine similarity
        xcs = F.cosine_similarity(x1[None,:,:], x2[:,None,:], dim=-1)
        # Ground truth labels
        target = torch.arange(x1.size(0))
  

        # Standard cross-entropy loss
        return F.cross_entropy(xcs.cuda() / temperature, target.cuda(), reduction="mean")
    def _infoNCE_loss(self, embeddings1, embeddings2, temperature=0.07):
        batch_size = embeddings1.shape[0]
        infonce_loss = self.loss_info(embeddings1, embeddings2,temperature)
        return infonce_loss

    def kl_loss1(self, thetas1, thetas2):
        print('thetas1',thetas1,' ',thetas1.shape,' ',thetas1[1].sum())
        theta_kld = F.kl_div(thetas1.log(), thetas2, reduction='sum')
        return theta_kld

    def _kl_loss2(self, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):
        # KL term
        # var division term
        var_division =posterior_variance / prior_variance
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = diff_means * diff_means / prior_variance
        # logvar det division term
        logvar_det_division = \
            prior_variance.log() - posterior_log_variance
        # combine terms
        KL = 0.5 * ((
            var_division + diff_term+ logvar_det_division).sum(1) - self.n_components )
        return KL

    def _rl_loss(self, true_word_dists, pred_word_dists):
        # Reconstruction term
        # print(true_word_dists.shape, true_word_dists, true_word_dists.sum(-1))
        # pdb.set_trace()
        #true_word_dists = F.softmax(true_word_dists, dim=-1)
        RL = -torch.sum(true_word_dists * torch.log(pred_word_dists + 1e-10), dim=1)
        return RL
    def export_beta_en(self,beta, vocab):
        beta=beta.cpu().detach().numpy()
        num_top_word = 15
        topic_str_list = file_utils.print_topic_words(beta, vocab, num_top_word=15)
        file_utils.save_text(topic_str_list, path=f'/home/ssliang/M3L-topic-model/output/ECNews_en_7-ablation.txt')
        return topic_str_list

    def export_beta_cn(self,beta, vocab):
        beta=beta.cpu().detach().numpy()
        num_top_word = 15
        topic_str_list = file_utils.print_topic_words(beta, vocab, num_top_word=15)
        file_utils.save_text(topic_str_list, path=f'/home/ssliang/M3L-topic-model/output/ECNews_cn_7--ablation.txt')
        return topic_str_list
    def export_beta_ja(self,beta, vocab):
        beta=beta.cpu().detach().numpy()
        num_top_word = 15
        topic_str_list = file_utils.print_topic_words(beta, vocab, num_top_word=15)
        file_utils.save_text(topic_str_list, path=f'/home/ssliang/M3L-topic-model/output/ECNews_ja_7--ablation.txt')
        return topic_str_list

    def beta_loss(self,beta0,beta1,k=15):
        assert k <= self.bow_size, "k must be <= input size."
        topics_all0 = np.zeros(4000)
        topics_all1 = np.zeros(4000)
        for i in range(self.n_components):
            _, idxs0 = torch.topk(beta0[i], k)
            idxs0=idxs0.cpu().detach().numpy()
            for j in idxs0:
                topics_all0[j]=1
        for i in range(self.n_components): 
            _, idxs1 = torch.topk(beta1[i], k)
            idxs1=idxs1.cpu().detach().numpy()
            for j in idxs0:
                topics_all1[j]=1
        loss0=topics_all0.sum()
        loss1=topics_all1.sum()
        loss=-(loss0+loss1)
        #print('loss',loss)
        return loss
    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_num, batch_samples in enumerate(loader):
            # batch_size x L x vocab_size
            X_bow1 = batch_samples['X_bow1']
            #print(X_bow1.shape)
            # print(X_bow1.shape, X_bow1[:,1,:], X_bow1[:,1,:].sum(dim=-1))
            # pdb.set_trace()
            X_bow1 = X_bow1.squeeze(dim=2)
            X_bow2 = batch_samples['X_bow2']
            X_bow2 = X_bow2.squeeze(dim=2)
            #print(X_bow2.shape)
            if self.USE_CUDA:
                X_bow1 = X_bow1.cuda()
                X_bow2 = X_bow2.cuda()
            # forward pass

            prior_mean, prior_sigma, posterior_mean1, posterior_sigma1, posterior_log_sigma1,\
            posterior_mean2, posterior_sigma2, posterior_log_sigma2, posterior_mean2_1, posterior_sigma2_1, posterior_log_sigma2_1, \
            posterior_mean3_1, posterior_sigma3_1, posterior_log_sigma3_1,word_dist_collect1,word_dist_collect2,word_dist_collect2_1,word_dist_collect3, \
            theta1, theta2, theta2_1, theta3_1=self.model(X_bow1,X_bow2)
            # backward pass
            model_size = sum(t.numel() for t in self.model.parameters())
            print("model size",model_size)
            # recon_losses for each language
            rl_loss1 = self._rl_loss(X_bow1[:,1,:], word_dist_collect1)
            rl_loss2 = self._rl_loss(X_bow1[:,0,:], word_dist_collect2)
            rl_loss3 = self._rl_loss(X_bow2[:,0,:], word_dist_collect2_1)
            rl_loss4 = self._rl_loss(X_bow2[:,1,:], word_dist_collect3)
            # KL between distributions of every language pair
            
            # InfoNCE loss/NTXentLoss
            beta=self.model.beta
            #kl_cross =  self.kl_loss1(theta1,theta2_new)+self.kl_loss1(theta2_1,theta3_1new)
            
            kl_cross1 = self._kl_loss2(prior_mean, prior_sigma,
                                 posterior_mean1, posterior_sigma1, posterior_log_sigma1)
            kl_cross2 = self._kl_loss2(prior_mean,prior_sigma,
                                 posterior_mean2, posterior_sigma2, posterior_log_sigma2)
            kl_cross3 = self._kl_loss2(prior_mean, prior_sigma,
                                 posterior_mean2_1, posterior_sigma2_1, posterior_log_sigma2_1)
            kl_cross4 = self._kl_loss2(prior_mean, prior_sigma,
                                 posterior_mean3_1, posterior_sigma3_1, posterior_log_sigma3_1)
            kl_cross=kl_cross1+kl_cross2+kl_cross3+kl_cross4
            
            
            infoNCE_cross = self._infoNCE_loss(theta1,theta2)
            infoNCE_cross += self._infoNCE_loss(theta2_1, theta3_1)
            rl_loss=rl_loss1+rl_loss2+rl_loss3+rl_loss4
            loss =10*kl_cross.mean()+infoNCE_cross*10+13*rl_loss.mean()
            print('rl',rl_loss.mean(),'kl',kl_cross.mean(),'info',infoNCE_cross)
            #list = [loss,rl_loss.mean(),kl_cross.mean(),infoNCE_cross,betaloss]
    
            #data = pd.DataFrame([list])
            #data.to_csv('./output/kl2_record_betaloss1,5_kl10¡ª¡ªtopic==10.csv', mode='a', header=False, index=False) 

            loss.backward()
            self.optimizer.step()
            # compute train loss
            samples_processed += X_bow1.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss

    def fit(self, train_dataset, validation_dataset, save_dir=None, verbose=True, patience=5, delta=0):
        """
        Train the CTM model.

        :param train_dataset: PyTorch Dataset class for training data.
        :param validation_dataset: PyTorch Dataset class for validation data. If not None, the training stops if validation loss doesn't improve after a given patience
        :param save_dir: directory to save checkpoint models to.
        :param verbose: verbose
        :param patience: How long to wait after last time validation loss improved. Default: 5
        :param delta: Minimum change in the monitored quantity to qualify as an improvement. Default: 0

        """
       

        self.model_dir = save_dir
        self.train_data = train_dataset
        self.validation_data = validation_dataset
        if self.validation_data is not None:
            self.early_stopping = EarlyStopping(patience=3, verbose=verbose, path=save_dir, delta=delta)
        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_data_loader_workers)

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        pbar = tqdm(self.num_epochs, position=0, leave=True)
        for epoch in range(self.num_epochs):
            print("-"*10, "Epoch", epoch+1, "-"*10)
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss, = self._train_epoch(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()
            pbar.update(1)

            if self.validation_data is not None:
                validation_loader = DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=True,
                                               num_workers=self.num_data_loader_workers)
                # train epoch
                s = datetime.datetime.now()
                val_samples_processed, val_loss, theta1,theta2_new,theta2_1,theta3_1new= self._validation(validation_loader)
                e = datetime.datetime.now()

                # report
                if verbose:
                    print("Epoch: [{}/{}]\tSamples: [{}/{}]\tValidation Loss: {}\tTime: {}".format(
                        epoch + 1, self.num_epochs, val_samples_processed,
                        len(self.validation_data) * self.num_epochs, val_loss, e - s))

                pbar.set_description("Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tValid Loss: {}\tTime: {}".format(
                    epoch + 1, self.num_epochs, samples_processed,
                    len(self.train_data) * self.num_epochs, train_loss, val_loss, e - s))
                """
                self.early_stopping(val_loss, self)
                if self.early_stopping.early_stop:
                    print("Early stopping")

                    break
                """
            else:
                # save last epoch
                #self.best_components = self.model.beta1
                self.best_components = self.model.beta
                if save_dir is not None:
                    self.save(save_dir)
            pbar.set_description("Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                epoch + 1, self.num_epochs, samples_processed,
                len(self.train_data) * self.num_epochs, train_loss, e - s))

            #print topics for every epoch
            #self.best_components=self.model.beta
            topic_cn=self.export_beta_cn(self.model.beta[0], self.vocab_cn)
            topic_en=self.export_beta_en(self.model.beta[1], self.vocab_en1)
            topic_ja=self.export_beta_ja(self.model.beta[2], self.vocab_ja)
            print('topic_cn',topic_cn)
            print('topic_en',topic_en)
            print('topic_ja',topic_ja)
            
        pbar.close()
        return theta1,theta2_new,theta2_1,theta3_1new

    def _validation(self, loader):
        """Validation epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            # batch_size x L x vocab_size
            X_bow1= batch_samples['X_bow1']
            X_bow1 = X_bow1.squeeze(dim=2)
            X_bow2= batch_samples['X_bow2']
            X_bow2 = X_bow2.squeeze(dim=2)
            if self.USE_CUDA:
                X_bow1 = X_bow1.cuda()
                X_bow2 = X_bow2.cuda()
            # forward pass
            self.model.zero_grad()
            prior_mean, prior_sigma, posterior_mean1, posterior_sigma1, posterior_log_sigma1,\
            posterior_mean2, posterior_sigma2, posterior_log_sigma2, posterior_mean2_1, posterior_sigma2_1, posterior_log_sigma2_1, \
            posterior_mean3_1, posterior_sigma3_1, posterior_log_sigma3_1,word_dist_collect1,word_dist_collect2,word_dist_collect2_1,word_dist_collect3, \
            theta1, theta2, theta2_1, theta3_1=self.model(X_bow1,X_bow2)
            # backward pass

            # recon_losses for each language
            rl_loss1 = self._rl_loss(X_bow1[:,1,:], word_dist_collect1)
            rl_loss2 = self._rl_loss(X_bow1[:,0,:], word_dist_collect2)
            rl_loss3 = self._rl_loss(X_bow2[:,0,:], word_dist_collect2_1)
            rl_loss4 = self._rl_loss(X_bow2[:,1,:], word_dist_collect3)
            # KL between distributions of every language pair
            
            # InfoNCE loss/NTXentLoss
            beta=self.model.beta
            #kl_cross =  self.kl_loss1(theta1,theta2_new)+self.kl_loss1(theta2_1,theta3_1new)
            
            kl_cross1 = self._kl_loss2(prior_mean, prior_sigma,
                                 posterior_mean1, posterior_sigma1, posterior_log_sigma1)
            kl_cross2 = self._kl_loss2(prior_mean,prior_sigma,
                                 posterior_mean2, posterior_sigma2, posterior_log_sigma2)
            kl_cross3 = self._kl_loss2(prior_mean, prior_sigma,
                                 posterior_mean2_1, posterior_sigma2_1, posterior_log_sigma2_1)
            kl_cross4 = self._kl_loss2(prior_mean, prior_sigma,
                                 posterior_mean3_1, posterior_sigma3_1, posterior_log_sigma3_1)
            kl_cross=kl_cross1+kl_cross2+kl_cross3+kl_cross4
            
            
            infoNCE_cross = self._infoNCE_loss(theta1,theta2)
            infoNCE_cross += self._infoNCE_loss(theta2_1, theta3_1)
            rl_loss=rl_loss1+rl_loss2+rl_loss3+rl_loss4
            loss =10*kl_cross.mean()+infoNCE_cross*10+13*rl_loss.mean()
            loss.backward()
            self.optimizer.step()
            samples_processed += X_bow1.size()[0]
            val_loss += loss.item()
        val_loss /= samples_processed

        return samples_processed, val_loss,theta1,theta2,theta2_1,theta3_1

    def get_thetas(self, dataset, n_samples=20):
        """
        Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via
        the parameter n_sample.

        :param dataset: a PyTorch Dataset containing the documents
        :param n_samples: the number of sample to collect to estimate the final distribution (the more the better).
        """
        return self.get_doc_topic_distribution(dataset, n_samples=n_samples)

    def get_doc_topic_distribution(self, dataset, n_samples=10):
        """
        Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via
        the parameter n_sample.

        :param dataset: a PyTorch Dataset containing the documents
        :param n_samples: the number of sample to collect to estimate the final distribution (the more the better).
        """
        
        
        #model_file =
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict)
        checkpoint['state_dict'].pop('topic_word_matrix')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_data_loader_workers)
        pbar = tqdm(n_samples, position=0, leave=True)
        final_thetas = []
        for sample_index in range(n_samples):
            with torch.no_grad():
                collect_theta = []

                for batch_samples in loader:
                    X_bow1 = batch_samples['X_bow1']
                    X_bow1 = X_bow1.squeeze(dim=1)
                    X_bow2 = batch_samples['X_bow2']
                    X_bow2 = X_bow2.squeeze(dim=1)
            
                    if self.USE_CUDA:
                      X_bow1 = X_bow1.cuda()
                      X_bow2 = X_bow2.cuda()
                    # forward pass
                    self.model.zero_grad()
                    thetas = self.model.get_theta(X_bow1, X_bow2)
                    collect_theta.extend(thetas.detach().cpu().numpy())

                pbar.update(1)
                pbar.set_description("Sampling: [{}/{}]".format(sample_index + 1, n_samples))

                final_thetas.append(np.array(collect_theta))
        pbar.close()
        return np.sum(final_thetas, axis=0) / n_samples


    def get_topics(self, k=15):
        """
        Retrieve topic words.
        :param k: int, number of words to return per topic, default 10.
        """
        assert k <= self.bow_size, "k must be <= input size."
        component_dists = self.best_components
        #print('component_dists:', component_dists.shape)
        topics_all = []
        for l in range(self.num_lang):
            topics = defaultdict(list)
            for i in range(self.n_components):
                _, idxs = torch.topk(component_dists[l][i], k)
                component_words = [self.train_data.idx2token[l][idx]
                                   for idx in idxs.cpu().numpy()]
                topics[i] = component_words
            topics_all.append(topics)
        return topics_all

  
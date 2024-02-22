import torch
from torch import nn
from torch.nn import functional as F

from networks.inference_bow import ContextualInferenceNetwork
from networks.mlp import MLPnetwork
import numpy as np

# ----- Multilingual  -----

class ContrastiveMultilingualDecoderNetwork(nn.Module):

    def __init__(self, input_size, bert_size, n_components, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, label_size=0, num_languages=2):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            learn_priors : bool, make priors learnable parameter
            num_languages: no. of languages in dataset
        """
        super(ContrastiveMultilingualDecoderNetwork, self).__init__()

        # input_size: same as vocab size
        self.input_size = input_size
        # n_components: no. of topics
        self.n_components = 10
        self.model_type = model_type
        self.hidden_sizes = 100
        self.activation = 'softplus'
        self.dropout = 0.2
        self.learn_priors = learn_priors

        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * self.n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        # prior_var is same for all languages
        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * self.n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.num_languages = num_languages
        # each language has their own inference network (assume num_lang=2 for now) ---- instantiate separate inference networks for each language
        self.inf_net1 = ContextualInferenceNetwork(input_size, bert_size, self.n_components, self.hidden_sizes, self.activation)
        self.inf_net2 = ContextualInferenceNetwork(input_size, bert_size, self.n_components, self.hidden_sizes, self.activation)

        # topic_word_matrix is K x V, where L = no. of languages
        self.topic_word_matrix = None

        # beta is L x K x V where L = no. of languages
        self.beta = torch.Tensor(num_languages, self.n_components, input_size)
       
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
            
        self.beta = nn.Parameter(self.beta.cuda())
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)
        self.MLPnetwork1 = MLPnetwork(20000)
    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        """Forward pass."""
        # x_bert: batch_size x L x bert_dim
        # print('DecoderNet - forward')
        # print('x_bert:', x_bert.shape)
        # pass to first x_bert to inference net1 (input is batch_size x bert_dim)
        #self.beta=self.beta.cuda()
        self.beta1_pred = torch.Tensor(self.n_components, self.input_size)
        #self.beta1_pred = nn.Parameter(self.beta1_pred.cuda())
        #print('beta',self.beta,' ',self.beta1_pred)
        self.beta1_pred =self.MLPnetwork1(self.beta[0])
        posterior_mu1, posterior_log_sigma1 = self.inf_net1(x[:, 0, :])
        posterior_sigma1 = torch.exp(posterior_log_sigma1)

        # pass to second x_bert to inference net2 (input is batch_size x bert_dim)
        posterior_mu2, posterior_log_sigma2 = self.inf_net2(x[:, 1, :])
        posterior_sigma2 = torch.exp(posterior_log_sigma2)

        # generate separate thetas for each language
        z1 = self.reparameterize(posterior_mu1, posterior_log_sigma1)
        z2 = self.reparameterize(posterior_mu2, posterior_log_sigma2)
        theta1 = F.softmax(z1, dim=1)
        theta2 = F.softmax(z2, dim=1)

        thetas_no_drop = torch.stack([theta1, theta2])
        z_no_drop = torch.stack([z1, z2])

        theta1 = self.drop_theta(theta1)
        theta2 = self.drop_theta(theta2)
        theta1 = theta1.cuda()
        theta2 = theta2.cuda()
        thetas = torch.stack([theta1, theta2])

        word_dist_collect = []
        word_dist_pred=[]
        for l in range(self.num_languages):
            # compute topic-word dist for language l
            # separate theta and separate beta for each language
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(thetas[l], self.beta[l])), dim=1)
            #print(word_dist.shape)
            word_dist_collect.append(word_dist)
        # word_dist_collect: L x batch_size x input_size
        word_dist_collect = torch.stack([w for w in word_dist_collect])
       
        self.topic_word_matrix = self.beta
        #print('beta:', self.beta.shape)
        return self.prior_mean, self.prior_variance, posterior_mu1, posterior_sigma1, posterior_log_sigma1, \
            posterior_mu2, posterior_sigma2, posterior_log_sigma2, word_dist_collect,thetas_no_drop,z_no_drop
"""
    def get_theta(self, x, x_bert, lang_index=0):
        with torch.no_grad():

            # we do inference PER LANGUAGE, so we use only 1 inference network at a time
            if lang_index == 0:
                posterior_mu, posterior_log_sigma = self.inf_net1(x, x_bert)
            else:
                posterior_mu, posterior_log_sigma = self.inf_net2(x, x_bert)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            #print('theta:', theta.shape)
            return theta
"""

# ----- Our class -----
class TrilingualDecoderNetwork(nn.Module):

    def __init__(self, input_size, bert_size, n_components, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, label_size=0, num_languages=3):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            learn_priors : bool, make priors learnable parameter
            num_languages: no. of languages in dataset
        """
        super(TrilingualDecoderNetwork, self).__init__()

        # input_size: same as vocab size
        self.input_size = input_size
        # n_components: no. of topics
        self.n_components = n_components # you can change it
        self.model_type = model_type
        self.hidden_sizes = (100,100)
        self.activation = 'softplus'
        self.dropout = 0.2
        self.learn_priors = learn_priors

      
        topic_prior_mean = 0.0
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)
        self.a = 1 * np.ones((1, int(self.n_components))).astype(np.float32)
        self.prior_mean = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T), requires_grad=False).cuda()
        self.prior_variance = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 /self.n_components))).T + (1.0 / (self.n_components * self.n_components)) * np.sum(1.0 / self.a, 1)).T),               requires_grad=False).cuda()
        self.num_languages = num_languages
        # each language has their own inference network (assume num_lang=2 for now) ---- instantiate separate inference networks for each language
        self.inf_net1 = ContextualInferenceNetwork(input_size, self.n_components, self.hidden_sizes, self.activation)
        self.inf_net2 = ContextualInferenceNetwork(input_size, self.n_components, self.hidden_sizes, self.activation)
        self.inf_net2_1 = ContextualInferenceNetwork(input_size, self.n_components, self.hidden_sizes, self.activation)
        self.inf_net3_1 = ContextualInferenceNetwork(input_size,  self.n_components, self.hidden_sizes, self.activation)
        self.inf_net1w = ContextualInferenceNetwork(input_size,  self.n_components, self.hidden_sizes, self.activation)
        self.inf_net2w = ContextualInferenceNetwork(input_size,  self.n_components, self.hidden_sizes, self.activation)
        self.inf_net2_1w = ContextualInferenceNetwork(input_size,  self.n_components, self.hidden_sizes, self.activation)
        self.inf_net3_1w = ContextualInferenceNetwork(input_size, self.n_components, self.hidden_sizes, self.activation)
        '''self.inf_net1w = load_state_dict()'''

        # topic_word_matrix is K x V, where L = no. of languages
        self.topic_word_matrix = None

        # beta is L x K x V where L = no. of languages
        self.beta = torch.Tensor(3, self.n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(nn.init.xavier_uniform_(self.beta))
        #self.beta.weight.requires_grad=True
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=True)
        self.beta_batchnorm.weight.requires_grad = False
        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)
        #print(self.input_size)
        self.F12 = MLPnetwork(self.input_size)
        self.F23 = MLPnetwork(self.input_size)
        self.F31w = MLPnetwork(self.input_size)
        self.F13w = MLPnetwork(self.input_size)
        self.F3w2w = MLPnetwork(self.input_size)
        self.F2w1w = MLPnetwork(self.input_size)
    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x1, x2):
        """Forward pass."""
        ## sample mu and theta
        posterior_mu1, posterior_log_sigma1 = self.inf_net1(x1[:, 1, :])  
        posterior_sigma1 = torch.exp(posterior_log_sigma1)

        # pass to second x_bert to inference net2 (input is batch_size x bert_dim)
        posterior_mu2, posterior_log_sigma2 = self.inf_net2(x1[:, 0, :])
        posterior_sigma2 = torch.exp(posterior_log_sigma2)
        
        posterior_mu2_1, posterior_log_sigma2_1 = self.inf_net2_1(x2[:, 0, :]
        posterior_sigma2_1 = torch.exp(posterior_log_sigma2_1)

        posterior_mu3_1, posterior_log_sigma3_1 = self.inf_net3_1(x2[:, 1, :])
        posterior_sigma3_1 = torch.exp(posterior_log_sigma3_1)
        # generate separate thetas for each language
        z1 = self.reparameterize(posterior_mu1, posterior_log_sigma1)
        z2 = self.reparameterize(posterior_mu2, posterior_log_sigma2)
        z2_1 = self.reparameterize(posterior_mu2_1, posterior_log_sigma2_1)
        z3_1 = self.reparameterize(posterior_mu3_1, posterior_log_sigma3_1)
        theta1 = F.softmax(z1, dim=1)
        theta2 = F.softmax(z2, dim=1)
        theta2_1 = F.softmax(z2_1, dim=1)
        theta3_1 = F.softmax(z3_1, dim=1)
        theta1 = theta1.cuda()
        theta2 = theta2.cuda()
        thetas01 = torch.stack([theta1, theta2])
        theta2_1 = theta2_1.cuda()
        theta3_1 = theta3_1.cuda()
        thetas02 = torch.stack([theta2_1, theta3_1])
        ## update theta2 and theta3_1 
        theta2_new = F.softmax(self.F12(torch.cat([theta1,theta2], dim=-1)),dim=1)
        theta3_1new = F.softmax(self.F23(torch.cat([theta2_1,theta3_1], dim=-1)),dim=1)
        self.topic_word_matrix = self.beta
        ## generate bow matrix
        word_dist_new1 = F.softmax(
                self.beta_batchnorm(torch.matmul(theta1, self.beta[0])), dim=1) # Bow1_tilde
        word_dist_new2 = F.softmax(
                self.beta_batchnorm(torch.matmul(theta2_new, self.beta[1])), dim=1)  # Bow2_tilde
        word_dist_new2_1 = F.softmax(
                self.beta_batchnorm(torch.matmul(theta2_1, self.beta[1])), dim=1)  # Bow2_1_tilde
        word_dist_new3 = F.softmax(
                self.beta_batchnorm(torch.matmul(theta3_1new, self.beta[2])), dim=1) # Bow3_tilde
        posterior_mu1w, posterior_log_sigma1w = self.inf_net1w(word_dist_new1)
        posterior_mu2w, posterior_log_sigma2w = self.inf_net2w(word_dist_new2)
        posterior_mu2_1w, posterior_log_sigma2_1w = self.inf_net2_1w(word_dist_new2_1)
        posterior_mu3w, posterior_log_sigma3w = self.inf_net3_1w(word_dist_new3)
        # regenerate pseudo thetas for each language
        z1w = self.reparameterize(posterior_mu1w, posterior_log_sigma1w)
        z2w = self.reparameterize(posterior_mu2w, posterior_log_sigma2w)
        z2_1w = self.reparameterize(posterior_mu2_1w, posterior_log_sigma2_1w)
        z3w = self.reparameterize(posterior_mu3w, posterior_log_sigma3w)
        theta1w = F.softmax(z1w, dim=1) 
        theta2w = F.softmax(z2w, dim=1)
        theta2_1w = F.softmax(z2_1w, dim=1)
        theta3w = F.softmax(z3w, dim=1)
        theta1w = theta1w.cuda()
        theta2w = theta2w.cuda()
        theta2_1w = theta2_1w.cuda()
        theta3w = theta3w.cuda()
        ## updata theta2_1w, theta3w, theta1w
        theta2_1w_new = F.softmax(self.F3w2w(torch.cat([theta3w,theta2_1w], dim=-1)), dim=1) 
        theta1w_new = F.softmax(self.F2w1w(torch.cat([theta2w,theta1w], dim=-1)), dim=1)
        theta3w_new = F.softmax(self.F13w(torch.cat([theta1,theta3w], dim=-1)), dim=1)
        theta1w_new = F.softmax(self.F31w(torch.cat([theta3_1new,theta1w_new], dim=-1)), dim=1)
        
        
        return self.prior_mean, self.prior_variance, posterior_mu1, posterior_sigma1, posterior_log_sigma1, \
            posterior_mu2, posterior_sigma2, posterior_log_sigma2, posterior_mu2_1, posterior_sigma2_1, posterior_log_sigma2_1, \
            posterior_mu3_1, posterior_sigma3_1, posterior_log_sigma3_1,word_dist_new1, word_dist_new2, word_dist_new2_1, word_dist_new3,\
            theta1, theta2_new, theta1w_new, theta2w, \
            theta2_1, theta3_1new, theta2_1w_new, theta3w_new


"""

# ------- Original -------
class DecoderNetwork(nn.Module):

    def __init__(self, input_size, bert_size, infnet, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, label_size=0):

        super(DecoderNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(n_components, int) and n_components > 0, \
            "n_components must be type int > 0."
        assert model_type in ['prodLDA', 'LDA'], \
            "model type must be 'prodLDA' or 'LDA'"
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.topic_word_matrix = None

        # print('hidden_sizes:', hidden_sizes)
        if infnet == "zeroshot":
            self.inf_net = ContextualInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size)
        elif infnet == "combined":
            self.inf_net = CombinedInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size)
        else:
            raise Exception('Missing infnet parameter, options are zeroshot and combined')

        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)

        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
       
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, x_bert, labels=None):
       
        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        theta = self.drop_theta(theta)

        # prodLDA vs LDA
        if self.model_type == 'prodLDA':
            # in: batch_size x input_size x n_components
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
            # word_dist: batch_size x input_size
            self.topic_word_matrix = self.beta
        elif self.model_type == 'LDA':
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            self.topic_word_matrix = beta
            word_dist = torch.matmul(theta, beta)
            # word_dist: batch_size x input_size
        else:
            raise NotImplementedError("Model Type Not Implemented")

        # classify labels

        estimated_labels = None

        if labels is not None:
            estimated_labels = self.label_classification(theta)

        return self.prior_mean, self.prior_variance, posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, estimated_labels

    def get_theta(self, x, x_bert, labels=None):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
            #posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta
"""
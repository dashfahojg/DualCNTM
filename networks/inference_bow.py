from collections import OrderedDict
from torch import nn
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualInferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, input_size,output_size, hidden_sizes,
                 activation='softplus', dropout=0.2, label_size=0):
        super(ContextualInferenceNetwork, self).__init__()
         
        self.fc11 = nn.Linear(input_size, 200)
        self.fc12 = nn.Linear(200, 200)
        self.fc21 = nn.Linear(200, output_size)
        self.fc22 = nn.Linear(200, output_size)

        self.fc1_drop = nn.Dropout(dropout)
        self.z_drop = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(output_size, affine=True)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(output_size, affine=True)
        self.logvar_bn.weight.requires_grad = False


    def forward(self, x):
        #print('x',x)
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_drop(e1)
        print('e1',e1.shape)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        return mu, logvar
        
from collections import OrderedDict
from torch import nn
import torch


class MLPnetwork(nn.Module):
        
      def __init__(self, input_size):
        
          super(MLPnetwork, self).__init__()
          self.net1 = nn.Sequential(nn.Linear(2*input_size,input_size),nn.BatchNorm1d(input_size),nn.ReLU())
          self.net2 = nn.Sequential(nn.Linear(input_size,input_size), nn.BatchNorm1d(input_size),nn.ReLU())
      def forward(self,theta):
          theta = self.net1(theta)
          theta = self.net2(theta)
          return theta
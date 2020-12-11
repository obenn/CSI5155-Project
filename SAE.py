# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
from torch import nn

from semisupervised import StackedAutoEncoderClassifier

class StackedAutoEncoder(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, encoder=None, decoder=None):
        super(StackedAutoEncoder, self).__init__()
        if encoder is None:
            assert input_dim is not None, "The input feature dimension should be inputed"
            assert output_dim is not None, "The number of classes should be added."
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 100),
                nn.ReLU(True),
                nn.Linear(100, 50),
                nn.ReLU(True),
                nn.Linear(50, output_dim))
        else:
            self.encoder = encoder
        if decoder is None:
            self.decoder = nn.Sequential(
                nn.Linear(output_dim, 50),
                nn.ReLU(True),
                nn.Linear(50, 100),
                nn.ReLU(True),
                nn.Linear(100, input_dim),
                nn.ReLU(True))
        else:
            self.decoder = decoder

    def init_parameters(self):
        """
        Some methods to initialize the paramethers of network
        Usage: network.apply(init_parameters)
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in
              name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in
              name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.orthogonal_(k)
        for k in b:
            nn.init.constant_(k, 0)

    def forward(self, x):
        """
        return encoder_x: hidden variable
        re_x: reconstruct input
        """
        encoder_x = self.encoder(x)
        reconstruct_x = self.decoder(encoder_x)
        return encoder_x, reconstruct_x

import os

import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from torchdrug import core, layers, utils, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.ProteinHybridNetwork")
class ProteinHybridNetwork(nn.Module, core.Configurable):
    output_dim = 1024

    def __init__(self, model1: nn.Module, model2: nn.Module):
        super(ProteinHybridNetwork, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.output_dim = self.model1.output_dim + self.model2.output_dim
        print(model1)
        print(model2)

    def forward(self, graph, input, all_loss=None, metric=None):
        model1_output = self.model1(graph, input, all_loss, metric)
        model2_output = self.model2(graph, input, all_loss, metric)
        graph_feature = torch.cat([model1_output["graph_feature"], model2_output["graph_feature"]], dim=1)
        residue_feature = torch.cat([model1_output["residue_feature"], model2_output["residue_feature"]], dim=1)
        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }


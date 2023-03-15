""" Interaction Block """
from typing import Optional, Dict, Callable

import torch

from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import TensorProduct, Linear, FullyConnectedTensorProduct

from data import AtomicDataDict
from nn.nonlinearities import ShiftedSoftPlus
from ._graph_mixin import GraphModuleMixin

class Attention(torch.nn.Module):
    def __init__(self, irreps_q, irreps_k, irreps_v):
        super().__init__()
        self.q = Linear(irreps_q, irreps_in)
        self.k = Linear(irreps_k, irreps_in)
        self.v = Linear(irreps_v, irreps_in)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, q, k, v):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = self.softmax(attn)
        output = torch.bmm(attn, v)
        return output
    

    
    
    

class NonlocalBlock(GraphModuleMixin, torch.nn.Module):
    avg_num_neighbors: Optional[float]
    use_sc: bool

    def __init__(
        self,
        irreps_in,
        irreps_out,
    ) -> None:
        """
        InteractionBlock.

        :param irreps_node_attr: Nodes attribute irreps
        :param irreps_edge_attr: Edge attribute irreps
        :param irreps_out: Output irreps, in our case typically a single scalar
        :param radial_layers: Number of radial layers, default = 1
        :param radial_neurons: Number of hidden neurons in radial function, default = 8
        :param avg_num_neighbors: Number of neighbors to divide by, default None => no normalization.
        :param number_of_basis: Number or Basis function, default = 8
        :param irreps_in: Input Features, default = None
        :param use_sc: bool, use self-connection or not
        """
        super().__init__()

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.NODE_FEATURES_KEY,
                AtomicDataDict.NODE_ATTRS_KEY,
            ],

            irreps_out={AtomicDataDict.NODE_FEATURES_KEY: irreps_out},
        )

        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc

        feature_irreps_in = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        feature_irreps_out = self.irreps_out[AtomicDataDict.NODE_FEATURES_KEY]


        # - Build modules -
        self.resmlp_q = ResiMLP(
            irreps_in=feature_irreps_in,
            irreps_out=feature_irreps_in,
            internal_weights=True,
            shared_weights=True,
            activation=activation, 
            zero_init=True
        )
        self.resmlp_k = ResiMLP(
            irreps_in=feature_irreps_in,
            irreps_out=feature_irreps_in,
            internal_weights=True,
            shared_weights=True,
            activation=activation, 
            zero_init=True
        )
        self.resmlp_v = ResiMLP(
            irreps_in=feature_irreps_in,
            irreps_out=feature_irreps_in,
            internal_weights=True,
            shared_weights=True,
            activation=activation, 
            zero_init=True
        )
        
        self.attention = Attention(irreps_in=feature_irreps_in, irreps_in=feature_irreps_in, irreps_in=feature_irreps_in)



    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:



        x = data[AtomicDataDict.NODE_FEATURES_KEY]
        
        q = self.resmlp_q(x)  # queries
        k = self.resmlp_k(x)  # keys
        v = self.resmlp_v(x)  # values
        
        x=self.attention(q, k, v)
        
        x = self.linear_2(x)
        

        data[AtomicDataDict.NODE_FEATURES_KEY] = x
        return data

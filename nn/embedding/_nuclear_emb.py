import torch
import torch.nn.functional as F

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from data import AtomicDataDict
from nn.modules.electron_configurations import electron_config
from .._graph_mixin import GraphModuleMixin
import torch.nn as nn

@compile_mode("script")
class NuclearChargeEmbedding(GraphModuleMixin, torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int = 87,
        num_features: int = 128,
        set_features: bool = True,
        irreps_in=None,
        use_concat: bool = True,
    ):
        super().__init__()
        self.num_types = num_types
        self.num_features = num_features
        self.set_features = set_features
        self.use_concat = use_concat
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_features, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)
        
        self.extra_embedding = nn.Embedding(num_types, num_features).float()
        
        self.one_hot = F.one_hot(torch.arange(0, num_types), num_classes=num_types).float().cuda()
        self.one_hot_embedding = nn.Linear(num_types, num_features, bias=False
        ).float()
        
        self.electron_config = torch.tensor(electron_config).cuda()
        self.config_embedding = nn.Linear(self.electron_config.size(1), num_features, bias=False
        ).float()
        
        self.linear1 = nn.Linear(num_features*3, num_features, bias=False
        ).float() 
        
        # TODO valence electrons
        

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        print(data.keys())
        nuclear_charges = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        
        self.extra_embedding = self.extra_embedding.to(device=nuclear_charges.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        self.one_hot_embedding = self.one_hot_embedding.to(device=nuclear_charges.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype) 
        self.config_embedding = self.config_embedding.to(device=nuclear_charges.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype) 
        
        if nuclear_charges.device.type == "cpu":  # indexing is faster on CPUs
            value_extra = self.extra_embedding(nuclear_charges)  
            value_one_hot = self.one_hot_embedding(self.one_hot)[nuclear_charges] 
            value_config = self.config_embedding(self.electron_config)[nuclear_charges]
            if self.use_concat:
                data[AtomicDataDict.NODE_ATTRS_KEY]=torch.cat((value_extra, value_one_hot, value_config), 1) 
                if self.set_features:    
                    data[AtomicDataDict.NODE_FEATURES_KEY]=torch.cat((value_extra, value_one_hot, value_config), 1) 
            else:
                data[AtomicDataDict.NODE_ATTRS_KEY] = self.extra_embedding(nuclear_charges) + self.one_hot_embedding(self.one_hot)[nuclear_charges] +  self.config_embedding(self.electron_config)[nuclear_charges]
                if self.set_features: 
                    data[AtomicDataDict.NODE_FEATURES_KEY] = self.extra_embedding(nuclear_charges) + self.one_hot_embedding(self.one_hot)[nuclear_charges] +  self.config_embedding(self.electron_config)[nuclear_charges]

        else:  # gathering is faster on GPUs
            # value_extra= torch.gather(
            #     self.extra_embedding, 0, nuclear_charges.view(-1, 1).expand(-1, self.num_features)
            # )
            
            value_extra = self.extra_embedding(nuclear_charges)   
            value_one_hot= torch.gather(
                self.one_hot_embedding(self.one_hot), 0, nuclear_charges.view(-1, 1).expand(-1, self.num_features)
            )
            value_config= torch.gather(
                self.config_embedding(self.electron_config), 0, nuclear_charges.view(-1, 1).expand(-1, self.num_features)
            )
            if self.use_concat: 
                data[AtomicDataDict.NODE_ATTRS_KEY]=torch.cat((value_extra, value_one_hot, value_config), 1)  
                if self.set_features:  
                    data[AtomicDataDict.NODE_FEATURES_KEY]=torch.cat((value_extra, value_one_hot, value_config), 1)
            else:
                data[AtomicDataDict.NODE_ATTRS_KEY] = self.extra_embedding(nuclear_charges) + self.one_hot_embedding(self.one_hot)[nuclear_charges] + self.config_embedding(self.electron_config)[nuclear_charges]
                if self.set_features:  
                    data[AtomicDataDict.NODE_FEATURES_KEY] = self.extra_embedding(nuclear_charges) + self.one_hot_embedding(self.one_hot)[nuclear_charges] + self.config_embedding(self.electron_config)[nuclear_charges]

        data[AtomicDataDict.NODE_ATTRS_KEY]=self.linear1(data[AtomicDataDict.NODE_ATTRS_KEY]) 
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY]=self.linear1(data[AtomicDataDict.NODE_FEATURES_KEY]) 

        return data


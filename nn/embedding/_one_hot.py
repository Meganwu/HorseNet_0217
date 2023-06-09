import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from data import AtomicDataDict
from nn.modules.electron_configurations import electron_config
from .._graph_mixin import GraphModuleMixin
import torch.nn as nn

@compile_mode("script")
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int = 87,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_types, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)
        
        self.one_hot_linear = nn.Linear(87, 87, bias=False
        ).float()
  
        self.electron_config = torch.tensor(electron_config)
        
        self.config_linear = nn.Linear(self.electron_config.size(1), 87, bias=False
        ).float()
        

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        one_hot = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_types
        ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        if 'cuda'  in type_numbers.device.type:
           self.electron_config = self.electron_config.cuda()        
        # config_one = self.config_linear(self.electron_config.cuda())[type_numbers]
        config_one = self.config_linear(self.electron_config)[type_numbers]
        one_hot_feature=self.one_hot_linear(one_hot)
        data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot_feature+config_one
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot_feature+config_one
        return data

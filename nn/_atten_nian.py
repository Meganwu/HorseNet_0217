from typing import Optional, List
import torch
import torch.nn.functional

from e3nn.o3 import Linear,Activation
from data import AtomicDataDict, AtomicDataset
from nn import GraphModuleMixin


class Attention(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
        irreps_out=None,
    ):
        super().__init__()
        self.field = field
        out_field = out_field if out_field is not None else field
        self.out_field = out_field
        if irreps_out is None:
            irreps_out = irreps_in[field]

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out={out_field: irreps_out},
        )
  
        self.linear_q=Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        )
        self.linear_k=Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        )
        self.linear_v=Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        )
        self.activation_q=Activation(self.irreps_out[out_field],[torch.abs])
        self.activation_k=Activation(self.irreps_out[out_field],[torch.abs])
        self.activation_q=Activation(self.irreps_out[out_field],[torch.abs])
        
    def forward(self, data: AtomicDataDict.Type, eps: float=1e-8) -> AtomicDataDict.Type:
        x=data[self.field]
        q=self.linear_q(x)
        q=self.activation_q(q)
        k=self.linear_k(x)
        k=self.activation_k(k)
        v=self.linear_v(x)
        v=self.activation_v(v)
        d=q.shape[-1]
        dot=q@k.T
        A=torch.exp((dot-torch.max(dot))/d**0.5)
        norm=torch.sum(A,dim=-1, keepdim=True)+eps
        data[self.field]=(A/norm)@v
        
        return data
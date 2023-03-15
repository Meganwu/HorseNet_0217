
from nn.embedding import OneHotAtomEncoding

from nn._atomwise import AtomwiseLinear
from e3nn.o3 import Irreps

from nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)

from nn._convnetlayer import ConvNetLayer
import torch
from data import AtomicData




class HorseNet(torch.nn.Module):
    def __init__(self, lmax=2, parity=1, basis_kwargs={'r_max': 3.5}, cutoff_kwargs={'r_max': 3.5}, irreps_in_4={'pos': '1x1o', 'edge_index': None, 'node_attrs': '87x0e', 'node_features': '32x0e', 'edge_attrs': '1x0e+1x1o+1x2e', 'edge_embedding': '8x0e'},irreps_out_4=Irreps('32x0e')):
        super(HorseNet, self).__init__()
        self.one_hot=OneHotAtomEncoding()
        self.irreps_edge_sh=Irreps.spherical_harmonics(lmax=lmax, p=-1 if parity else 1)
        self.sphericalharmedge=SphericalHarmonicEdgeAttrs(irreps_edge_sh=irreps_edge_sh)
        self.radiabasis=RadialBasisEdgeEncoding(basis_kwargs=basis_kwargs,cutoff_kwargs=cutoff_kwargs)
        self.linear_1=AtomwiseLinear(irreps_in={'pos': '1x1o', 'edge_index': None, 'node_attrs': '87x0e', 'node_features': '87x0e', 'edge_attrs': '1x0e+1x1o+1x2e',
 'edge_embedding': '8x0e'},irreps_out=Irreps('32x0e'))

        self.convnet_1=ConvNetLayer(irreps_in=irreps_in_4, feature_irreps_hidden='32x0e+32x1e+32x2e+32x0o+32x1o+32x2o')
        self.convnet_2=ConvNetLayer(irreps_in=m5.irreps_out, feature_irreps_hidden='32x0e+32x1e+32x2e+32x0o+32x1o+32x2o')
        self.linear_2=AtomwiseLinear(irreps_in=m6.irreps_out,irreps_out=Irreps('16x0e'))
        self.linear_3=AtomwiseLinear(irreps_in=m7.irreps_out,irreps_out=Irreps('2x0e'))      

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
        d1=self.one_hot(data)
        d2=self.sphericalharmedge(d1)
        d3=self.radiabasis(d2)
        d4=self.linear_1(d3)
        d5=self.convnet_1(d4)
        d6=self.convnet_2(d5)
        d7=self.linear_2(d6)
        d8=self.linear_3(d7)
        data=d8
        return data

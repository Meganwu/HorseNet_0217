import os
from ase.io import read, write
import sys

sys.path.append('/scratch/work/wun2/github/HorseNet_0217')
import numpy as np
# from data.AtomicData import neighbor_list_and_relative_vec
import torch
import torch.utils.data as Data
import torch
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

from data import AtomicDataDict
from data import AtomicData
from utils.torch_geometric import Batch, Dataset

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import Config
from data import dataset_from_config


from utils.save_metric import save_loss_metrics
from data.preprocess import preprocess

from model._eng import SimpleIrrepsConfig, EnergyModel
from model._grads import ForceOutput, PartialForceOutput, StressForceOutput
from model._scaling import RescaleEnergyEtc, PerSpeciesRescale

from nequip.nn.embedding._one_hot_spin import OneHotSpinEncoding
from nequip.nn.embedding._one_hot_charge import OneHotChargeEncoding



class HorseNet(torch.nn.Module):
    def __init__(self, lmax=2, parity=1, basis_kwargs={'r_max': 3.5}, cutoff_kwargs={'r_max': 3.5}, irreps_in_4={'pos': '1x1o', 'edge_index': None, 'node_attrs': '87x0e', 'node_features': '32x0e', 'edge_attrs': '1x0e+1x1o+1x2e', 'edge_embedding': '8x0e'},irreps_out_4=Irreps('32x0e')):
        super(HorseNet, self).__init__()
        self.one_hot=OneHotAtomEncoding()
        self.irreps_edge_sh=Irreps.spherical_harmonics(lmax=lmax, p=-1 if parity else 1)
        self.sphericalharmedge=SphericalHarmonicEdgeAttrs(irreps_edge_sh=self.irreps_edge_sh)
        self.radiabasis=RadialBasisEdgeEncoding(basis_kwargs=basis_kwargs,cutoff_kwargs=cutoff_kwargs)
        self.linear_1=AtomwiseLinear(irreps_in={'pos': '1x1o', 'edge_index': None, 'node_attrs': '87x0e', 'node_features': '87x0e', 'edge_attrs': '1x0e+1x1o+1x2e',
 'edge_embedding': '8x0e'},irreps_out=Irreps('32x0e'))
        self.convnet_1=ConvNetLayer(irreps_in=irreps_in_4, feature_irreps_hidden='32x0e+32x1e+32x2e+32x0o+32x1o+32x2o')
        self.convnet_2=ConvNetLayer(irreps_in=self.convnet_1.irreps_out, feature_irreps_hidden='32x0e+32x1e+32x2e+32x0o+32x1o+32x2o')
        self.linear_2=AtomwiseLinear(irreps_in=self.convnet_2.irreps_out,irreps_out=Irreps('16x0e'))
        self.linear_3=AtomwiseLinear(irreps_in=self.linear_2.irreps_out,irreps_out=Irreps('2x0e'))      

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data['pos']=data['pos'].requires_grad_(True) 
        data=self.one_hot(data)
        data=self.sphericalharmedge(data)
        data=self.radiabasis(data)
        data=self.linear_1(data)
        data=self.convnet_1(data)
        data=self.convnet_2(data)
        data=self.linear_2(data)
        data=self.linear_3(data)
        data['atomic_energy'] = data['node_features'][:, 0]
        data['atomic_charges'] = data['node_features'][:, 1] 
        data['total_energy']=torch.zeros(data['batch']+1).scatter_add_(0, data['batch'], data['atomic_energy'])
        print(data['total_energy'], data['pos'])
        
        grad = torch.autograd.grad(
            [data['total_energy']],
            [data['pos']],
            create_graph=True,  # needed to allow gradients of this output during training
        )[0]
        
        data['forces'] = -grad

        return data
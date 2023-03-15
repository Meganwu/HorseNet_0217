import sys

sys.path.append('/scratch/work/wun2/github/HorseNet_0217')

from nn.embedding import OneHotAtomEncoding

from nn._atomwise import AtomwiseLinear
from e3nn.o3 import Irreps

from nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)

from nn._convnetlayer import ConvNetLayer

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
        data['total_energy']=torch.zeros(list(data['batch'][-1])+1).scatter_add_(0, data['batch'], data['atomic_energy'])
        print(data['total_energy'], data['pos'])

        grad = torch.autograd.grad(
            [data['total_energy']],
            [data['pos']],
            create_graph=True,  # needed to allow gradients of this output during training
        )[0]

        data['forces'] = -grad

        return data

#config = Config.from_file('configs/config.yaml')
#config.device='cuda'
#config['dataset_file_name']='inputs/nian_889_charge.extxyz'
#dataset_param = dataset_from_config(config, prefix="dataset")
#m1=SimpleIrrepsConfig(config)
#model=EnergyModel(config, initialize=True, dataset = dataset_param)
#model=PerSpeciesRescale(model, config, True, dataset=dataset_param)
#model=ForceOutput(model)
#model=RescaleEnergyEtc(model, config, True, dataset=dataset_param)

data_xyz=read('/scratch/phys/sin/Eric_summer_project_2022/Nian_calculations/process_OUTCAT_nian/nian_889_charge.extxyz',format='extxyz',index=':')
dataset=[AtomicData.from_ase(data_xyz[i],3.5) for i in range(len(data_xyz))]


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=Batch.from_data_list)
val_dataloader = DataLoader(val_dataset, batch_size=50, shuffle=True, collate_fn=Batch.from_data_list)
model=HorseNet().cuda()
loss_fn=torch.nn.MSELoss(reduction='mean').cuda()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001,amsgrad=True)
components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }

result_path='results_nequip_species_3'
try:
    os.mkdir(result_path)
except:
    pass
train_output=os.path.join(result_path,'output_physnet_charge_train.txt')
val_output=os.path.join(result_path,'output_physnet_charge_val.txt')
val_output_total=os.path.join(result_path,'output_physnet_charge_val_total.txt')
model_params=os.path.join(result_path,'physnet_charge_%s_param.pkl')
with open(train_output, 'a') as f:
    f.write('epoch, loss_e, loss_f, loss_qa, '+ ', '.join([i+'_f_mae' for i in components.keys()])+ ', '+', '.join([i+'_qa_mae' for i in components.keys()])+'\n')
with open(val_output, 'a') as f:
    f.write('epoch, loss_e, loss_f, loss_qa, '+ ', '.join([i+'_f_mae' for i in components.keys()])+ ', '+', '.join([i+'_qa_mae' for i in components.keys()])+'\n')
with open(val_output_total, 'a') as f:
    f.write('epoch, loss_e, loss_f, loss_qa, '+ ', '.join([i+'_f_mae' for i in components.keys()])+ ', '+', '.join([i+'_qa_mae' for i in components.keys()])+'\n')




epoches=10000
for epoch in tqdm(range(epoches)): 
    for train_i, train_d in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()
        data_t=train_d.to_dict()
        data_t['atom_types']=data_t['atomic_numbers']
        data_t={k:v.cuda() for k, v in data_t.items()}
        Z=data_t['atomic_numbers']
        Eref=data_t['total_energy']
        Fref=data_t['forces']
        Qaref=data_t['atomic_charges']
        pred = model(data_t)
        energy=pred['total_energy']
        forces=pred['forces']
        qa=pred['atomic_charges']
        print('aaaaaaaa', energy.shape, Eref.shape, forces.shape, Fref.shape, qa.shape, Qaref.shape)    
        loss_t_e=loss_fn(energy,Eref)
        loss_t_f=loss_fn(forces,Fref)
        loss_t_q=loss_fn(qa, Qaref)
        loss_t=loss_t_e+loss_t_f   
        loss_t.backward()
        print('loss_t:', loss_t.detach().cpu().numpy())
        optimizer.step()
        save_loss_metrics(id_n=train_i, atomic_n=Z, ref_e=Eref,ref_f=Fref,ref_qa=Qaref,pred_e=energy, pred_f=forces, pred_qa=qa, components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }, outfile=train_output) 
    for val_i, val_d in enumerate(val_dataloader):
        model.eval()
        data_v=val_d.to_dict()
        data_v['atom_types']=data_v['atomic_numbers']
        data_v={k:v.cuda() for k, v in data_v.items()}
        Z_v=data_v['atomic_numbers']
        Eref_v=data_v['total_energy']
        Fref_v=data_v['forces']
        Qaref_v=data_v['atomic_charges']
        pred_v = model(data_v)
        energy_v=pred_v['total_energy']
        forces_v=pred_v['forces']
        qa_v=pred_v['atomic_charges']    
        loss_v_e=loss_fn(energy_v,Eref_v)
        loss_v_f=loss_fn(forces_v,Fref_v)
        loss_v_q=loss_fn(qa_v, Qaref_v)
        loss_v=loss_v_e+loss_v_f   
        print('loss_v:', loss_v.detach().cpu().numpy())
        save_loss_metrics(id_n=val_i, atomic_n=Z_v, ref_e=Eref_v,ref_f=Fref_v,ref_qa=Qaref_v,pred_e=energy_v, pred_f=forces_v, pred_qa=qa_v, components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }, outfile=val_output,use_charge=True)
        
        
    
            
        
        

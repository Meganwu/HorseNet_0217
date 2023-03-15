import torch
import numpy as np
import os
from ase.io import read, write
import torch
import torch.utils.data as Data
import torch
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

from data import AtomicData, AtomicDataDict
from utils.torch_geometric import Batch, Dataset

from torch.utils.data import DataLoader

# convet AtomicDict to input of PhysNet and SpookyNet

def preprocess(batch_data):
        Z=batch_data['atomic_numbers'].squeeze()
        R=batch_data['pos']
#        R=torch.tensor(batch_data['pos'], requires_grad=True)
        idx_i=batch_data['edge_index'][0]
        idx_j=batch_data['edge_index'][1]
        batch_seg=batch_data['batch'].squeeze()
        Eref=batch_data['total_energy'].squeeze()
        Fref=batch_data['forces']
        Qaref=batch_data['atomic_charges'].squeeze()
        Qref=batch_data['total_charge'].squeeze()
        Sref=batch_data['total_charge'].squeeze()
        cell=batch_data['cell']
        cell_offsets=batch_data['edge_cell_shift']
        edge_shift=batch_data['edge_shift']
        num_batch=(batch_seg[-1]+1).tolist()
        return Z, R, idx_i, idx_j, batch_seg, Eref, Fref, Qaref, Qref, Sref, cell, cell_offsets, num_batch, edge_shift



class NormalizationTransformation:
    def __init__(self, dataset, partial_charge=True):
        e_total=np.array([dataset[i].total_energy.tolist()[0] for i in range(len(dataset))])
        mean = np.mean(e_total)
        std = np.std(e_total)
        
        self.e_shift=-mean
        self.e_scale=1/std
        self.partial_charge=partial_charge
        print('Energy shift: ', self.e_shift, 'Energy scale: ', self.e_scale)
        if self.partial_charge:
            q_total=np.concatenate([dataset[i].atomic_charges.numpy() for i in range(len(dataset))])
            mean_q = np.mean(q_total)
            std_q = np.std(q_total)
            self.q_shift=-mean_q
            self.q_scale=1/std_q
            
            print('Charge shift: ', self.q_shift, 'Charge scale: ', self.q_scale)
        
    def transform(self, dataset):
        for i in range(len(dataset)):
            dataset[i].total_energy=(dataset[i].total_energy+self.e_shift)*self.e_scale
            dataset[i].forces=dataset[i].forces*self.e_scale
            if self.partial_charge: 
               dataset[i].atomic_charges=(dataset[i].atomic_charges+self.q_shift)*self.q_scale
        return dataset
    def inverse_transform(self, dataset):  # dataset is a list of AtomicDataDict
        dataset['total_energy']=dataset['total_energy']/self.e_scale-self.e_shift
        dataset['forces']=dataset['forces']/self.e_scale
        if self.partial_charge:
            dataset['atomic_charges']=dataset['atomic_charges']/self.q_scale-self.q_shift
        return dataset


class LoadData():
    def __init__(self, filename='/scratch/phys/sin/Eric_summer_project_2022/Nian_calculations/process_OUTCAT_nian/nian_889_charge.extxyz', ratio=0.8, r_cut=3.5, index=':') -> None:
        self.filename=filename
        self.ratio = ratio
        self.r_cut=r_cut
        self.index=index

    
    def dataloader(self, normalize=True, val_batch_size=50, train_batch_size=5, shuffle=True):
        
        data_xyz=read(self.filename,format='extxyz',index=self.index)
        self.dataset=[AtomicData.from_ase(data_xyz[i],self.r_cut) for i in range(len(data_xyz))]
        self.size=len(self.dataset)
        if normalize:
            self.normal_model=NormalizationTransformation(self.dataset)
            self.dataset=self.normal_model.transform(self.dataset)
        
        self.train_size = int(self.ratio*len(self.dataset))+1
        self.val_size = len(self.dataset) - self.train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size])
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=Batch.from_data_list)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=True, collate_fn=Batch.from_data_list)
        return self.train_dataloader, self.val_dataloader
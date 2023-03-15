#!/usr/bin/env python
# coding: utf-8

from  utils import * 
import pickle
import pandas as pd
import os
import re
import lzma

import numpy as np

from ase.visualize import view
from ase import Atoms
from ase.io import read, write

from tqdm import tqdm

# screening dataset

#oc20_data=retrieval_info(filename='oc20_data_mapping.pkl')
#oc22_metadata=retrieval_info(filename='oc22_metadata.pkl')
#
## Find adsorbate-catalyst system-ids matching the filter criteria
## cu_oc20=oc20_data[oc20_data['bulk_symbols']=='Cu']
## cu_oc22=oc22_metadata[oc22_metadata['bulk_symbols']=='Cu']
#
#all_elements=list(oc20_data['bulk_symbols'].value_counts().keys().values)
#random_ids=[]
#
#for i in tqdm(all_elements):
#    a=oc20_data[oc20_data['bulk_symbols']==i].index
#    random_num=np.random.randint(0, high=len(a))
#    random_id=oc20_data.iloc[a[random_num]]['Unnamed: 0']
#    random_ids.append(random_id)
#
#sel_list=random_ids
#sel_list=',|'.join(sel_list)+','  # change format
#
#path_name='/scratch/phys/sin/Eric_summer_project_2022/Data_OCP/s2ef_all/s2ef_train_all/s2ef_train_all'
#s2ef_train_all=collect_dir_info(path_name, sel_list)
#s2ef_train_all=pd.DataFrame(s2ef_train_all)
#s2ef_train_all.to_csv('s2ef_train_all_select.csv')

s2ef_train_all=pd.read_csv('/scratch/work/wun2/github/HorseNet_0217/input_generate/s2ef_train_all_select.csv')

#conf_ids=[]

#for i in list(s2ef_train_all['2'].value_counts().index):
#    a0=s2ef_train_all[s2ef_train_all['2']==i].index[0]
#    conf_ids.append(a0)

#pd.DataFrame(conf_ids).to_csv('conf_ids.csv')
#conf_ids=conf_ids[0:1500]
conf_ids=pd.read_csv('conf_ids.csv')
conf_ids=conf_ids['0'].tolist()[3001:]
d1_1460=s2ef_train_all.loc[conf_ids]
sel_xyzs=[]
data_locs=d1_1460
data_locs=data_locs.reset_index(drop=True) 
for i in tqdm(range(len(data_locs))):

        sel_xyz=get_xyz(data_locs.loc[i])
        sel_xyzs.append(sel_xyz)


# Save extxyz
write('data_3001.extxyz',sel_xyzs,format='extxyz')



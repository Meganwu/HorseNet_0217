import os
from ase.io import read, write
import sys
sys.path.append('/scratch/work/wun2/github/HorseNet_0217')
import numpy as np
from data.AtomicData import neighbor_list_and_relative_vec
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
from tqdm import tqdm

from utils import Config
from data import dataset_from_config


from utils.save_metric import save_loss_metrics, loss_metrics, save_loss_total, generate_savefile
from data.preprocess import preprocess,NormalizationTransformation,LoadData

from model._eng import SimpleIrrepsConfig, EnergyModel
from model._grads import ForceOutput, PartialForceOutput, StressForceOutput
from model._scaling import RescaleEnergyEtc, PerSpeciesRescale

from torch_ema import ExponentialMovingAverage
from utils import instantiate
from train.loss import Loss, LossStat
from train.metrics import Metrics


class Model:
    def __init__(self, model_name):
        self.model_name=model_name
        
    def HorseNet_1(self):
        from model._eng import SimpleIrrepsConfig, EnergyModel
        from model._grads import ForceOutput, PartialForceOutput, StressForceOutput
        from model._scaling import RescaleEnergyEtc, PerSpeciesRescale
        config = Config.from_file('/scratch/work/wun2/github/HorseNet_0217/test_all/configs/config.yaml')
        config['dataset_file_name']='/scratch/work/wun2/github/HorseNet_0217/test_all/inputs/param_dataset.extxyz'
        dataset_param = dataset_from_config(config, prefix="dataset")
        config.device='cpu'
        m1=SimpleIrrepsConfig(config)
        model=EnergyModel(config, initialize=True, dataset = dataset_param)
        model=PerSpeciesRescale(model, config, True, dataset=dataset_param)
        model=ForceOutput(model)
        return model
    def HorseNet_2(self):
        from model._horsenet import HorseNet
        model=HorseNet()
        
        return model
    
    def SpookyNet(self):
        sys.path.append('/scratch/work/wun2/github/SpookyNet/')
        from spookynet import SpookyNet
        model=SpookyNet().float()
        return model
    
    def PhysNet(self):
        sys.path.append('/scratch/work/wun2/github/PhysNet_DER')
        from Neural_Net_evid import PhysNet, gather_nd, softplus_inverse

        model=PhysNet().float().cuda() 
        
    def forward(self):
        if self.model_name=='HorseNet_1':
            model=self.HorseNet_1()
            
        elif self.model_name=='HorseNet_2':
            model=self.HorseNet_2()
            
        elif self.model_name=='SpookyNet':
            model=self.SpookyNet()
            
        elif self.model_name=='PhysNet':
            model=self.PhysNet()
            
        return model
        


if __name__=='__main__':
    _remove_from_model_input={'atomic_charges',
    'atomic_energy',
    'forces',
    'partial_forces',
    'stress',
    'total_charge',
    'total_energy',
    'virial'}

    model=Model('HorseNet_1').forward().cuda()

    data=LoadData()
    train_dataloader, val_dataloader=data.dataloader()

    train_output, val_output, val_output_total, model_params, metric_val, metric_train=generate_savefile()

    loss_fn=torch.nn.MSELoss(reduction='mean').cuda()
    loss_fn_sum=torch.nn.MSELoss(reduction='sum').cuda()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005, weight_decay=0.0, amsgrad=False)
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor = 0.7)

    model_ema = ExponentialMovingAverage(model.parameters(), decay=0.99, use_num_updates=True)

    components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }


 # training   
    epoches=10000
    for epoch in tqdm(range(epoches)): 
        for train_i, train_d in enumerate(train_dataloader):
            optimizer.zero_grad()
            model.train()
            data_t=train_d.to_dict()
            data_t['atom_types']=data_t['atomic_numbers']
            data_t={k:v.cuda() for k, v in data_t.items()}
            Eref=data_t['total_energy']
            Fref=data_t['forces']
            Qaref=data_t['atomic_charges'] 
            
            data_t_input={k:v for k, v in data_t.items()}

            Z=data_t['atomic_numbers']
            pred = model(data_t_input)

            energy=pred['total_energy']
            forces=pred['forces']
            qa=pred['atomic_charges'] 
            # loss_t, loss_contrib_t = loss_final(pred=pred, ref=data_t)
  #          loss_t=loss_metrics(loss_fn, atomic_n=Z, ref_e=Eref,ref_f=Fref,ref_qa=Qaref,pred_e=energy, pred_f=forces, pred_qa=qa, components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }, use_charge=False, use_force=True, use_compenents_loss=False)  
            #loss_t_e=loss_fn_sum(energy,Eref)/Fref.shape[0]
            loss_t_e=loss_fn(energy,Eref)
            loss_t_f=loss_fn(forces,Fref)
            loss_t_q=loss_fn(qa, Qaref)
            loss_t=loss_t_e+loss_t_f+loss_t_q
            print('transformer_losssss', loss_t)
            loss_t.backward()
            optimizer.step()
            lr_scheduler.step(loss_t)
            model_ema.update(model.parameters())

   
            
            # data_t_inverse=normal_model.inverse_transform(data_t)
            # pred_t_inverse=normal_model.inverse_transform(pred)
            data_t_inverse=data.normal_model.inverse_transform(data_t)
            pred_t_inverse=data.normal_model.inverse_transform(pred)
            
            # batch_losses_t = loss_stat(loss_t.cpu(), loss_contrib_t)
            # batch_metrics_t = metrics(pred=pred_t_inverse, ref=data_t_inverse)
            # save_loss_total(train_i, batch_metrics_t, outfile=metric_train)
            # print('aaaa_train',list(batch_metrics_t.values())[-2]*list(batch_metrics_t.values())[-2]/batch_losses_t['loss'])
            energy_t=pred_t_inverse['total_energy']
            forces_t=pred_t_inverse['forces']
            qa_t=pred_t_inverse['atomic_charges']
            Eref_t=data_t_inverse['total_energy']
            Fref_t=data_t_inverse['forces']
            Qaref_t=data_t_inverse['atomic_charges']
            
       #     loss_t_e_inv=loss_fn_sum(energy_t,Eref_t)/Fref.shape[0]
            loss_t_e_inv=loss_fn(energy_t,Eref_t)
            loss_t_f_inv=loss_fn(forces_t,Fref_t)
            loss_t_q_inv=loss_fn(qa_t, Qaref_t)
            # print('loss_ttt', loss_t)
            print('aaaa_train',loss_t_e_inv, loss_t_f_inv, loss_t_q_inv)
            # print('aaaa_train_mae',(energy-Eref).mean())
            # print('aaaa_train_batch_metrics_t',batch_metrics_t)
            save_loss_metrics(id_n=train_i, atomic_n=Z, ref_e=Eref_t,ref_f=Fref_t,ref_qa=Qaref_t,pred_e=energy_t, pred_f=forces_t, pred_qa=qa_t, components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }, outfile=train_output)
        if epoch%5==1:
            torch.save(model.state_dict(),model_params % epoch)
            loss_val_sum=[]
            loss_val_e_sum=[]
            loss_val_f_sum=[]
            loss_val_q_sum=[]

            for val_i, val_d in enumerate(val_dataloader):
                optimizer.zero_grad()
                model.eval()
                data_v=val_d.to_dict()
                data_v['atom_types']=data_v['atomic_numbers']
                data_v={k:v.cuda() for k, v in data_v.items()}
                
                data_unscaled_v = data_v


                input_data_v = {
                k: v
                for k, v in data_unscaled_v.items() if k not in _remove_from_model_input
                }   
                # with torch.no_grad(): 
                pred_v = model(input_data_v) 
                del input_data_v
                Z_v=pred_v['atomic_numbers']
                
                
                data_v_inverse=data.normal_model.inverse_transform(data_v)
                pred_v_inverse=data.normal_model.inverse_transform(pred_v)
            #    Eref_v=data_unscaled_v1['total_energy']
            #    Fref_v=data_unscaled_v1['forces']
            #    Qaref_v=data_unscaled_v1['atomic_charges']
            
                # with torch.no_grad():
                Eref_v=data_v_inverse['total_energy']
                Fref_v=data_v_inverse['forces']
                Qaref_v=data_v_inverse['atomic_charges']

                energy_v=pred_v_inverse['total_energy']
                forces_v=pred_v_inverse['forces']
                qa_v=pred_v_inverse['atomic_charges']    
            
                loss_v_e=loss_fn_sum(energy_v,Eref_v)/Fref_v.shape[0]
                loss_v_f=loss_fn(forces_v,Fref_v)
                loss_v_q=loss_fn(qa_v, Qaref_v)
        
                
                # loss_val_sum.append(loss_v.cpu().detach().numpy().item())
                loss_val_e_sum.append(loss_v_e.cpu().detach().numpy().item())
                loss_val_f_sum.append(loss_v_f.cpu().detach().numpy().item())
                loss_val_q_sum.append(loss_v_q.cpu().detach().numpy().item())
                save_loss_metrics(id_n=val_i, atomic_n=Z_v, ref_e=Eref_v,ref_f=Fref_v,ref_qa=Qaref_v,pred_e=energy_v, pred_f=forces_v, pred_qa=qa_v, components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }, outfile=val_output,use_charge=True)

        #               print("####epoch %s    loss_v:  %s   loss_v_e:  %s   loss_v_f:  %s  loss_q: %s #####" % (val_i,loss_v.cpu().detach().numpy().item(), loss_v_e.cpu().detach().numpy().item(), loss_v_f.cpu().detach().numpy().item(), loss_v_q.cpu().detach().numpy().item()))
   
            loss_val_e_avg=np.array(loss_val_e_sum).mean()
            loss_val_f_avg=np.array(loss_val_f_sum).mean()
            loss_val_q_avg=np.array(loss_val_q_sum).mean()              
            with open(val_output_total, 'a') as f:
                f.write('Epoch:%s, %s, %s, %s \n' %  (epoch,  loss_val_e_avg.item(), loss_val_f_avg.item(), loss_val_q_avg.item())) 
            print("####epoch %s    loss_e:  %s   loss_f:  %s  loss_q: %s #####" % (epoch, loss_val_e_avg.item(), loss_val_f_avg.item(), loss_val_q_avg.item()))               








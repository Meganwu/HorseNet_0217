import torch
import os
import numpy as np


def generate_savefile(result_path='nequip_loss_results', components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }):

    try:
        os.mkdir(result_path)
        
    except:
        pass
    train_output=os.path.join(result_path,'output_charge_train.txt')
    val_output=os.path.join(result_path,'output_charge_val.txt')
    val_output_total=os.path.join(result_path,'output_charge_val_total.txt')
    model_params=os.path.join(result_path,'physnet_charge_%s_param.pkl')
    metric_val=os.path.join(result_path,'metrics_val.txt')
    metric_train=os.path.join(result_path,'metrics_train.txt')
    with open(train_output, 'a') as f:
        f.write('epoch, loss_e, loss_atomic_e, loss_f, loss_qa, '+ ', '.join([i+'_f_mae' for i in components.keys()])+ ', '+', '.join([i+'_qa_mae' for i in components.keys()])+'\n')
    with open(val_output, 'a') as f:
        f.write('epoch, loss_e, loss_atomic_e, loss_f, loss_qa, '+ ', '.join([i+'_f_mae' for i in components.keys()])+ ', '+', '.join([i+'_qa_mae' for i in components.keys()])+'\n')
    with open(val_output_total, 'a') as f:
        f.write('epoch, loss_e, loss_atomic_e, loss_f, loss_qa, '+ ', '.join([i+'_f_mae' for i in components.keys()])+ ', '+', '.join([i+'_qa_mae' for i in components.keys()])+'\n')

    with open(metric_train, 'a') as f:
        f.write('epoch, loss_e, loss_atomic_e, loss_f, loss_qa, '+ ', '.join([i+'_f_mae' for i in components.keys()])+ ', '+', '.join([i+'_qa_mae' for i in components.keys()])+'\n')

    with open(metric_val, 'a') as f:
        f.write('epoch, loss_e, loss_atomic_e, loss_f, loss_qa, '+ ', '.join([i+'_f_mae' for i in components.keys()])+ ', '+', '.join([i+'_qa_mae' for i in components.keys()])+'\n')
    return train_output, val_output, val_output_total, model_params, metric_val, metric_train

def loss_metrics(loss_fn, atomic_n=None, ref_e=None,ref_f=None,ref_qa=None,pred_e=None, pred_f=None, pred_qa=None, components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }, use_charge=True, use_force=True, use_compenents_loss=True):
    loss_fn=torch.nn.MSELoss(reduction='mean').to(device=ref_e.device)
#    linear_param=torch.nn.Linear(7,1, bias=False).to(device=ref_e.device)
    loss_total_e=loss_fn(pred_e,ref_e)
    if use_force:
        loss_total_f=loss_fn(pred_f,ref_f)
    if use_charge:
        loss_total_qa=loss_fn(pred_qa, ref_qa)
    loss_f_specie={}
    loss_qa_specie={}
    try:
        atomic_n_repeat=torch.repeat_interleave(atomic_n.unsqueeze(1), 3, dim=1)
        assert atomic_n_repeat.shape==pred_f.shape
    except:
        atomic_n_repeat=torch.repeat_interleave(atomic_n, 3, dim=1)
        assert atomic_n_repeat.shape==pred_f.shape
    for i, j in components.items():
        if len(pred_f[atomic_n_repeat==j])>0:
            if use_force:
                loss_f_specie[i]=loss_fn(pred_f[atomic_n_repeat==j],ref_f[atomic_n_repeat==j])
            if use_charge:
                loss_qa_specie[i]=loss_fn(pred_qa[atomic_n==j],ref_qa[atomic_n==j])
        else:
            loss_f_specie[i]=torch.tensor(0,device=ref_e.device, dtype=torch.float32)
            loss_qa_specie[i]=torch.tensor(0,device=ref_e.device, dtype=torch.float32)
    components_coeff={'H':2, 'C':2, 'N':2, 'O':2, 'Cu':2 }
    components_coeff={k: torch.tensor(v, dtype=torch.float32, device=ref_e.device) for k, v in components_coeff.items()}
    loss_total=loss_total_e
    if use_charge:
        loss_total=loss_total+loss_total_qa
    if use_compenents_loss:
        for i,j in components_coeff.items():
            loss_total=loss_total+loss_f_specie[i]*j
    else:
        loss_total=loss_total+loss_total_f
#    loss_total=torch.cat((loss_total_e.view(1), loss_f_specie['H'].view(1), loss_f_specie['C'].view(1), loss_f_specie['N'].view(1), loss_f_specie['O'].view(1), loss_f_specie['Cu'].view(1), loss_total_qa.view(1)))
#    loss_total=linear_param(loss_total)

    return loss_total

def save_loss_metrics(id_n=None, atomic_n=None, ref_e=None,ref_f=None,ref_qa=None,pred_e=None, pred_f=None, pred_qa=None, components={'H':1, 'C':6, 'N':7, 'O':8, 'Cu':29 }, outfile='outfile', use_charge=True):
    loss_fn=torch.nn.MSELoss(reduction='mean')
    loss_fn_sum=torch.nn.MSELoss(reduction='sum')
    loss_total_atomic_e=loss_fn_sum(pred_e,ref_e).cpu().detach().numpy().item()/ref_f.shape[0]
    loss_total_e=loss_fn(pred_e,ref_e).cpu().detach().numpy().item()
    loss_total_f=loss_fn(pred_f,ref_f).cpu().detach().numpy().item()
    if use_charge:
        loss_total_qa=loss_fn(pred_qa, ref_qa).cpu().detach().numpy().item()
    loss_f_specie={}
    loss_qa_specie={}
    try:
        atomic_n_repeat=torch.repeat_interleave(atomic_n.unsqueeze(1), 3, dim=1)
        assert atomic_n_repeat.shape==pred_f.shape
    except:
        atomic_n_repeat=torch.repeat_interleave(atomic_n, 3, dim=1)
        assert atomic_n_repeat.shape==pred_f.shape

    for i, j in components.items():
        if len(pred_f[atomic_n_repeat==j])>0:
            loss_f_specie[i]=loss_fn(pred_f[atomic_n_repeat==j],ref_f[atomic_n_repeat==j]).cpu().detach().numpy().item()
            if use_charge:
                loss_qa_specie[i]=loss_fn(pred_qa[atomic_n==j],ref_qa[atomic_n==j]).cpu().detach().numpy().item()
        else:
            loss_f_specie[i]=0
            loss_qa_specie[i]=0

    with open(outfile, 'a') as f:
        if use_charge:
            f.write('%s, %s, %s, %s, %s, %s, %s \n' %  (id_n, loss_total_e, loss_total_atomic_e, loss_total_f, loss_total_qa, ', '.join([str(loss_f_specie[i]) for i in components.keys()]), ', '.join([str(loss_qa_specie[i]) for i in components.keys()])))
        else:
            f.write('%s, %s, %s, %s, %s \n' %  (id_n, loss_total_e, loss_total_atomic_e, loss_total_f, ', '.join([str(loss_f_specie[i]) for i in components.keys()])))
    return loss_f_specie, loss_qa_specie


def save_loss_total(id_n, batch_metrics, outfile='metrics_val.csv'):
    loss_q_mae= list(batch_metrics.values())[0].detach().cpu().numpy()[0]
    loss_q_rmse= list(batch_metrics.values())[1].detach().cpu().numpy()[0] 
    loss_q_atomic_mae= list(batch_metrics.values())[2].detach().cpu().tolist()
    loss_q_atomic_mae =  ', '.join([str(i) for i in loss_q_atomic_mae])    

    
    loss_f_mae= list(batch_metrics.values())[3].detach().cpu().numpy()[0]
    loss_f_rmse= list(batch_metrics.values())[4].detach().cpu().numpy()[0] 
    loss_f_atomic_mae= list(batch_metrics.values())[5].detach().cpu().tolist()
    loss_f_atomic_mae =  ', '.join([str(i) for i in loss_f_atomic_mae])    
    loss_f_atomic_rmse= list(batch_metrics.values())[6].detach().cpu().tolist()
    loss_f_atomic_rmse =  ', '.join([str(i) for i in loss_f_atomic_rmse]) 
    
    loss_e_mae= list(batch_metrics.values())[7].detach().cpu().numpy()[0]
    loss_e_rmse= list(batch_metrics.values())[8].detach().cpu().numpy()[0] 
    
    with open(outfile, 'a') as f:
        f.write('%s, %s, %s, %s, %s \n' %  (id_n, loss_e_mae, loss_f_mae, loss_q_mae, loss_f_atomic_mae))

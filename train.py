import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from copy import deepcopy
from FMDA.utils import print_state
from FMDA.hparams import set_hparams
from FMDA.datasets import myDataset
from FMDA import algorithms
from FMDA.evaluate import evaluate_std_overall, evaluate_std_split_overall,  evaluate_loss_mat, evaluate_EO_mat, evaluate_loss_mat_split,evaluate_loss_mat_Ind
from FMDA.evaluate import evaluate_acc
from FMDA import datasets
torch.use_deterministic_algorithms(True)

def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    print(f"Random seed set as {seed}")

if __name__ == "__main__":
    ################################################
    ################ Basic Settings ################
    ################################################

    ################ Processing Inputs ################
    parser = argparse.ArgumentParser(description='FL')
    parser.add_argument('--data_dir', type=str, default="./FMDA/data/")
    parser.add_argument('--dataset', type=str, default="Support")
    parser.add_argument('--algorithm', type=str, default="FedAvg")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--A_C_alpha', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--n_commu', type=int, default=50)
    parser.add_argument('--epoch_per_commu', type=int, default=2)
    parser.add_argument('--n_step', type=int, default= 100)
    parser.add_argument('--momentum_beta', type=float, default=0)
    parser.add_argument('--momentum_lambda', type=float, default=1)
    parser.add_argument('--rad', type=float, default=0)
    parser.add_argument('--n_client', type=int, default=2)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    
    args = parser.parse_args()
    hparams = set_hparams(args)
    '''
    if args.algorithm == "Global":
        hparams["n_client"] = 1
    '''
    hparams['alpha'] = args.alpha
    hparams['gamma'] = args.gamma
    hparams['A_C_alpha'] = args.A_C_alpha
    hparams['rad'] = args.rad
    hparams['n_client'] = args.n_client
    hparams['n_step'] = args.n_step
    hparams['lr'] = args.lr
    def save_checkpoint(filename, algorithm):
        save_dict = {
            "args": vars(args),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))
    print_state(args, hparams)


    ################ Initialize Parameters ################
    algorithm_dict = None

    seed_torch(args.seed)
    checkpoint_vals = collections.defaultdict(lambda: [])
    
    
    
    ################ Initialize Classes ################
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(args.dataset)
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, hparams)
    else:
        raise NotImplementedError
        
    # set the index
    index = -1  
    if index == -1:
        num_splits = len(dataset.total_te[1].unique())
    else:
        num_splits = len(dataset.total_te[0][:,index].unique())
    
    
    total_test = myDataset(dataset.total_te[0],dataset.total_te[1],num_splits)
    client_tr = []
    client_te = []
    total_len = len(total_test)
    for i in range(hparams['n_client']):
        train_temp = myDataset(dataset.tr_datasets[i][0],dataset.tr_datasets[i][1],num_splits)
        client_tr.append(train_temp)
        test_temp = myDataset(dataset.te_datasets[i][0],dataset.te_datasets[i][1],num_splits)
        client_te.append(test_temp)
        total_len = total_len + len(train_temp) + len(test_temp)
    #### for global testing datasets
    from copy import deepcopy
    global_hparams = deepcopy(hparams)
    if args.dataset in vars(datasets):
        for i in range(hparams['n_client']):
            if i == 0:
                global_merge_x = dataset.tr_datasets[i][0]
                global_merge_y = dataset.tr_datasets[i][1]
                global_merge_x_te = dataset.te_datasets[i][0]
                global_merge_y_te = dataset.te_datasets[i][1]
            else:
                global_merge_x = torch.concat([global_merge_x, dataset.tr_datasets[i][0]], axis = 0)
                global_merge_y = torch.concat([global_merge_y, dataset.tr_datasets[i][1]], axis = 0)
                global_merge_x_te = torch.concat([global_merge_x_te, dataset.te_datasets[i][0]], axis = 0)
                global_merge_y_te = torch.concat([global_merge_y_te, dataset.te_datasets[i][1]], axis = 0)
        
        global_te = [myDataset(global_merge_x_te,global_merge_y_te,num_splits)]
        global_tr = [myDataset(global_merge_x,global_merge_y,num_splits)]
    else:
        raise NotImplementedError
    ####
    
    
    client_list = []
    server = None
    test_server = None
    if args.algorithm in vars(algorithms):
        if args.algorithm == "FairFed" or args.algorithm == "DIPOLE":
            server = vars(algorithms)[args.algorithm](dataset.input_shape, dataset.num_classes, hparams, client_tr, is_server = -1)
        else:
            server = vars(algorithms)[args.algorithm](dataset.input_shape, dataset.num_classes, hparams, is_server = -1)
        n_client = hparams['n_client']
        if args.algorithm == "Global":
            n_client = 1
        if args.algorithm == "FairFed"or args.algorithm == "DIPOLE":
            for i in range(n_client):
                client_list.append(vars(algorithms)[args.algorithm](dataset.input_shape, dataset.num_classes, hparams, client_tr, is_server = i))
        else:
            for i in range(n_client):
                client_list.append(vars(algorithms)[args.algorithm](dataset.input_shape, dataset.num_classes, hparams, is_server = i))
    else:
        raise NotImplementedError

    #########################################################
    ################ Training and Evaluation ################
    #########################################################

    ################ Initialize Parameters ################
    output_results = {'acc_mat':[],'acc_mat_overall':[],'loss_total':[], 'model':[]}
    start_step = 0
    start_commu = 0
    n_commu = args.n_commu
    epoch_per_commu = args.epoch_per_commu
    
    for commu in range(n_commu):
        # train the clients
        
        n_client = hparams['n_client']
        if args.algorithm == "Global":
            n_client = 1
        for indiv in range(n_client):
            step_per_epoch = int(len(client_tr[indiv]) / hparams['batch_size'])
            if len(client_tr[indiv]) > 0 and not len(client_tr[indiv]) == hparams['batch_size'] * step_per_epoch:
                step_per_epoch = step_per_epoch + 1
            client_step = 0
            is_stop_local_train = 0
            for epoch in range(max(epoch_per_commu, 100)):
                
                if args.algorithm == "Global":
                    
                    train_loaders = DataLoader(global_tr[indiv], batch_size = hparams['batch_size'], shuffle=True)
                else:
                    train_loaders = DataLoader(client_tr[indiv], batch_size = hparams['batch_size'], shuffle=True)
                train_minibatches_iterator = iter(train_loaders)
                results_step = []
                for step in range(step_per_epoch):
                    client_step += 1
                    X, Y = next(train_minibatches_iterator)
                    results_step = client_list[indiv].update(X, Y)
                    if client_step > args.n_step:
                        is_stop_local_train = 1
                        break
                if is_stop_local_train == 1:
                    break
        # communication
        if args.algorithm == "FMDA":
            lambda_mat = server.server_update(client_list, client_tr)
            for indiv in range(hparams['n_client']):
                client_list[indiv].set_lambda_mat(lambda_mat)
        elif args.algorithm == "FMDA_M_N":
            lambda_mat_A, lambda_mat_C = server.server_update(client_list, client_tr)
            for indiv in range(hparams['n_client']):
                client_list[indiv].set_lambda_mat(lambda_mat_A, lambda_mat_C)
        else:
            if args.algorithm == "FairFed" or args.algorithm == "DIPOLE":
                ACC_list = []
                EO_list = []
                for indiv, c in enumerate(client_list):
                    EO, AP, worst_TPR, overall_acc = evaluate_EO_mat(client_tr[indiv], c, hparams)
                    EO_list.append(EO) 
                    acc = evaluate_acc(client_tr[indiv], c, hparams)
                    ACC_list.append(acc)
                server.server_update(client_list, client_tr, EO_list, ACC_list)
            else:
                server.server_update(client_list, client_tr)
        # check if modified in the server
        
        # evaluation
        if args.algorithm == "Split":
            
            acc_mat, acc_mat_overall, number_mat = evaluate_std_split_overall(client_te, client_list, hparams)
            
            loss_mat = evaluate_loss_mat_split(client_te, client_list, hparams)
        else:
            
            acc_mat, acc_mat_overall, number_mat = evaluate_std_overall(client_te, server, hparams)
                
            loss_mat = evaluate_loss_mat(client_te, server, hparams)
                
        output_results['acc_mat'].append(acc_mat)
        output_results['acc_mat_overall'].append(acc_mat_overall)
        
        output_results['loss_total'].append(loss_mat)
        if args.algorithm == "Split":
            client_stat_list = []
            for client_i in client_list:
                model_temp = deepcopy(client_i.state_dict())
                client_stat_list.append(model_temp)
                
            output_results['model'].append(client_stat_list)
        else:
            model_temp = deepcopy(server.state_dict())
            output_results['model'].append(model_temp)

        save_checkpoint(f'model_epoch{commu}.pkl', server)
    
    ################ Output Result ################
    # model selection
    # the most simple
    m_id = 4
    loss_total = output_results['loss_total'][4].sum()
    for i in range(len(output_results['loss_total'])):
        temp = output_results['loss_total'][i].sum()
        if temp < loss_total and i >= 4:
            m_id = i
            loss_total = temp
    
    if hparams['test_EO'] == 1:
        if args.algorithm == "FairFed" or args.algorithm == "DIPOLE":
            test_server = vars(algorithms)[args.algorithm](dataset.input_shape, dataset.num_classes, hparams, client_tr, is_server = -1)
        else:
            test_server = vars(algorithms)[args.algorithm](dataset.input_shape, dataset.num_classes, hparams, is_server = -1)
        if args.algorithm == "Split":
            EO = 0
            AP = 0
            worst_TPR = 0
            overall_acc = 0
            for server_i in output_results["model"][m_id]:
                
                test_server.load_state_dict(server_i)
                
                EO_temp, AP_temp, worst_TPR_temp, overall_acc_temp = evaluate_EO_mat(global_te[0], test_server, hparams)
                EO = EO + EO_temp * 1.0/len(output_results["model"][m_id])
                AP = AP + AP_temp * 1.0/len(output_results["model"][m_id])
                worst_TPR = worst_TPR + worst_TPR_temp * 1.0/len(output_results["model"][m_id])
                overall_acc = overall_acc + overall_acc_temp * 1.0/len(output_results["model"][m_id])
                
                
        else:
            
            test_server.load_state_dict(output_results["model"][m_id])
            EO, AP, worst_TPR, overall_acc = evaluate_EO_mat(global_te[0], test_server, hparams)
        print("EO: ", EO, " AP:", AP)
        print("worst_TPR:", worst_TPR, "overall:", overall_acc)
    
    ###############################
    # averaged
    print("m_id:",m_id)
    def cal_result(output_acc, output_acc_overall):
        avg_std_attr = 0 
        client_acc_vector = torch.zeros(output_acc.shape[0])
        for i in range(output_acc.shape[0]):
            client_acc_vector[i] = (output_acc[i,:] * number_mat[i,:] / number_mat[i,:].sum()).sum()
    
        avg_mean = client_acc_vector.mean()
        worst_attr = 1
        acc_attr_list = torch.zeros(output_acc.shape[1])
        for i in range(output_acc.shape[1]):
            print(output_acc[:,i])
            acc_temp = output_acc[:,i] * number_mat[:,i] / (number_mat[:,i].sum())
            acc_temp = acc_temp.sum()
            worst_temp = acc_temp
            if worst_temp < worst_attr:
                worst_attr = worst_temp
            acc_attr_list[i] = acc_temp
        avg_std_attr = acc_attr_list.std()
        
        avg_std_client = 0
        worst_client = 1
        acc_client_list = torch.zeros(output_acc.shape[0])
        for i in range(output_acc.shape[0]):
            acc_temp = output_acc[i,:] * number_mat[i,:] / (number_mat[i,:].sum())
            acc_temp = acc_temp.sum()
            worst_temp = acc_temp
            if worst_temp < worst_client:
                worst_client = worst_temp
            acc_client_list[i] = acc_temp
        avg_std_client = acc_client_list.std()
        ###############################
        # overall acc
        print("m_id:",m_id)
        avg_mean_overall = 0
        print(output_acc_overall)
        for i in range(output_acc_overall.shape[0]):
            for j in range(output_acc_overall.shape[1]):
                avg_mean_overall += output_acc_overall[i,j] * number_mat[i,j]/number_mat.sum() 
                
        return avg_std_attr, avg_std_client, avg_mean, worst_attr, worst_client, avg_mean_overall
    if args.algorithm == "Split":
        avg_std_attr, avg_std_client, avg_mean, worst_attr, worst_client, avg_mean_overall = 0, 0, 0, 0, 0, 0
        for temp_mat in output_results["acc_mat"][m_id]:
            temp_avg_std_attr, temp_avg_std_client, temp_avg_mean, temp_worst_attr, temp_worst_client, temp_avg_mean_overall = cal_result(temp_mat, temp_mat)
            avg_std_attr += temp_avg_std_attr/len(output_results["acc_mat"][m_id])
            avg_std_client += temp_avg_std_client/len(output_results["acc_mat"][m_id])
            avg_mean += temp_avg_mean/len(output_results["acc_mat"][m_id])
            worst_attr += temp_worst_attr/len(output_results["acc_mat"][m_id])
            worst_client += temp_worst_client/len(output_results["acc_mat"][m_id])
            avg_mean_overall += temp_avg_mean_overall/len(output_results["acc_mat"][m_id])
    else:
        avg_std_attr, avg_std_client, avg_mean, worst_attr, worst_client, avg_mean_overall = cal_result(output_results["acc_mat"][m_id], output_results["acc_mat_overall"][m_id])

    print("avg_std_attr:", avg_std_attr)
    print("avg_std_client:", avg_std_client)
    print("avg_mean:", avg_mean) 
    
    print("avg worstcase attr acc:", worst_attr)
    print("avg worstcase client acc:", worst_client)
    print("avg_mean_overall:", avg_mean_overall) 
    
    
    df = pd.DataFrame(output_results["acc_mat"][m_id])
    df.to_csv("./FMDA/results/"+str(args.algorithm)+"_"+str(args.dataset) 
              + "_gamma" + str(hparams['gamma']) + "_alpha" + str(hparams["alpha"]) 
              + "_nc" + str(hparams["n_client"]) + "_epoPerCom" + str(hparams["epoch_per_commu"]) + ".csv")
    

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

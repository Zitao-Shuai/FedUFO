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
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from copy import deepcopy
from FMDA.utils import print_state
from FMDA.hparams import set_hparams
from FMDA.datasets import myDataset
import FMDA.datasets
import FMDA.algorithms

def evaluate_acc(test_data, model, hparams):
    step_per_epoch = int(len(test_data) / hparams['batch_size'])
    
    if len(test_data) > 0:
        if len(test_data) > 0 and not len(test_data) == hparams['batch_size'] * step_per_epoch:
            step_per_epoch = step_per_epoch + 1
        test_loaders = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle=True)
        test_minibatches_iterator = iter(test_loaders)
        sum = 0
        total_num = len(test_data)
        for step in range(step_per_epoch):
            X, Y = next(test_minibatches_iterator)
            Y_hat = model.predict(X)
        
            Y_hat = torch.argmax(Y_hat, dim = 1)
        
            for i in range(Y.shape[0]):
                if Y_hat[i] == Y[i]:
                    sum = sum + 1
        
        acc = 1.0 * sum / total_num
        return acc    
    else:
        # this combination is null return a negative number to identify
        return -1

def evaluate_std(te_dataset, server, hparams, mode = None):
    # evaluate the performance of the client models
    # calculate the matrix

    acc_mat = torch.zeros((len(te_dataset),server.num_classes))
    for id_te, te in enumerate (te_dataset):#id_te == id_client

        attr_list = te.split_by_attr()
        for id_attr, attr in enumerate(attr_list):
            
            acc = evaluate_acc(attr, server, hparams)
            acc_mat[id_te, id_attr] = acc

    return acc_mat

def evaluate_std_overall(te_dataset, server, hparams, mode = None):
    # evaluate the performance of the client models
    # calculate the matrix

    acc_mat = torch.zeros((len(te_dataset),server.num_classes))
    acc_mat_overall = torch.zeros((len(te_dataset),server.num_classes))
    number_mat = torch.zeros((len(te_dataset),server.num_classes))
    
    for id_te, te in enumerate (te_dataset):#id_te == id_client

        attr_list, number_vector = te.split_by_attr_overall()
        
        for k in range(server.num_classes):
            number_mat[id_te, k] = number_vector[k]

        
        for id_attr, attr in enumerate(attr_list):
            
            acc = evaluate_acc(attr, server, hparams)
            acc_mat[id_te, id_attr] = acc
            
            acc_mat_overall[id_te, id_attr] = acc 


    return acc_mat, acc_mat_overall, number_mat

def evaluate_loss(test_data, model, hparams, mode = None):
    
    step_per_epoch = int(len(test_data) / hparams['batch_size'])
    if len(test_data) > 0:
        if len(test_data) > 0 and not len(test_data) == hparams['batch_size'] * step_per_epoch:
            step_per_epoch = step_per_epoch + 1
        test_loaders = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle=True)
        test_minibatches_iterator = iter(test_loaders)
        loss_sum = 0
        for step in range(step_per_epoch):
            X, Y = next(test_minibatches_iterator)
            Y_hat = model.predict(X)
            loss = F.cross_entropy(Y_hat, Y)
            loss_sum = loss_sum + loss / step_per_epoch
    
        return loss_sum
    else:
        return 0

def evaluate_loss_mat(te_dataset, server, hparams, mode = None):
    # evaluate the performance of the client models

    loss_mat = torch.zeros((len(te_dataset),server.num_classes))
    
    for id_te, te in enumerate (te_dataset):#id_te == id_client

        attr_list = te.split_by_attr()
        for id_attr, attr in enumerate(attr_list):
            loss = evaluate_loss(attr, server, hparams)
            loss_mat[id_te, id_attr] = loss
    return loss_mat


def evaluate_EO(test_data, model, hparams, mode = None):
    sens_index = hparams['sens_index']
    step_per_epoch = int(len(test_data) / hparams['batch_size'])
    if len(test_data) > 0:
        if len(test_data) > 0 and not len(test_data) == hparams['batch_size'] * step_per_epoch:
            step_per_epoch = step_per_epoch + 1
        test_loaders = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle=True)
        test_minibatches_iterator = iter(test_loaders)
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        sum5 = 0
        sum6 = 0
        sum7 = 0
        sum8 = 0
        sum = 0
        correct = 0
        sum_a0 = 0
        sum_a1 = 0
        correct_a0 = 0
        correct_a1 = 0
        for step in range(step_per_epoch):
            X, Y = next(test_minibatches_iterator)
            Y_hat = model.predict(X)
            
            for i in range(Y_hat.shape[0]):

                sum += 1
                if Y[i] == 0 and X[i,sens_index] == 0:
                    sum1 = sum1 + 1
                    sum_a0 += 1
                    if Y_hat[i].argmax() == 0:
                        sum2 = sum2 + 1
                        correct += 1
                        correct_a0 += 1
                
                if  Y[i]== 0 and X[i,sens_index] == 1:
                    sum3 = sum3 + 1
                    sum_a1 += 1
                    if Y_hat[i].argmax() == 0:
                        sum4 = sum4 + 1
                        correct += 1
                        correct_a1 += 1

                if  Y[i]== 1 and X[i,sens_index] == 0:
                    sum5 = sum5 + 1
                    sum_a0 += 1
                    if Y_hat[i].argmax() == 1:
                        sum6 = sum6 + 1
                        correct += 1
                        correct_a0 += 1

                if  Y[i]== 1 and X[i,sens_index] == 1:
                    sum7 = sum7 + 1
                    sum_a1 += 1
                    if Y_hat[i].argmax() == 1:
                        sum8 = sum8 + 1
                        correct += 1
                        correct_a1 += 1
        
        if sum5 == 0:
            temp3 = 0
        else:
            temp3 = (sum6 * 1.0 / sum5)

        if sum7 == 0:
            temp4 = 0
        else:
            temp4 = (sum8 * 1.0 / sum7)
        
        EO = abs(temp3 - temp4)
        AP = abs(correct_a0/sum_a0 - correct_a1/sum_a1)
        worst_TPR = min(temp3, temp4)
        overall_acc = correct / sum
        return EO, AP, worst_TPR, overall_acc
    else:
        return 0,0
def evaluate_EO_mat(te, server, hparams, mode = None):
    
    EO, AP, worst_TPR, overall_acc = evaluate_EO(te, server, hparams)
    if EO < 0:
        EO = -EO   
    return EO, AP, worst_TPR, overall_acc

def evaluate_std_split_client(te_dataset, server, hparams, idc):
    # don't test the dataset owned by client itself
    # evaluate the performance of the client models
    acc_mat = torch.zeros((len(te_dataset),server.num_classes))
    number_mat = torch.zeros((len(te_dataset),server.num_classes))
    for id_te, te in enumerate (te_dataset):#id_te == id_client
        attr_list,number_vector = te.split_by_attr_overall()
        for k in range(server.num_classes):
            number_mat[id_te, k] = number_vector[k]
        for id_attr, attr in enumerate(attr_list):
            
            acc = evaluate_acc(attr, server, hparams)
            acc_mat[id_te, id_attr] = acc

    return acc_mat, number_mat

def evaluate_std_split_overall(te_dataset, client_list, hparams):
    acc_mat = []
    for index, client in enumerate(client_list):
        temp_acc_mat, number_mat = evaluate_std_split_client(te_dataset, client, hparams, index)
        acc_mat.append(temp_acc_mat) # averaged by clients
    return acc_mat, acc_mat, number_mat

def evaluate_loss_mat_split(te_dataset, client_list, hparams):
    loss_mat = torch.zeros((len(te_dataset),client_list[0].num_classes))
    for client in client_list:
        loss_mat = loss_mat + 1.0 * evaluate_loss_mat(te_dataset, client, hparams) / len(client_list) # averaged by clients
    return loss_mat

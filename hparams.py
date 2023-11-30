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
from FedUFO.utils import set_up
from copy import deepcopy

def set_hparams(args = None):
    ### initialize the hyperparameters
    
    # for SEER
    hparams = {"alpha": 1} # for LDA
    hparams["n_client"] = 2 # for client
    hparams['holdout_fraction'] = args.holdout_fraction
    hparams['dataset'] = args.dataset
    hparams['algorithm'] = args.algorithm
    hparams['alpha'] = args.alpha
    hparams['agnostic_alpha'] = args.agnostic_alpha
    hparams['test_EO'] = args.test_EO
    hparams['test_Analysis'] = args.test_Analysis
    hparams['agnostic'] = 0 
    if args.dataset == "SEER":
        hparams['batch_size'] = 128 # small/large
    else:
        hparams['batch_size'] = 32 # small/large
    if args.alpha == None:
        if args.dataset == "Support":
            hparams['alpha'] = 0.1
        elif args.dataset == "cardio":
            hparams['alpha'] = 10000
        elif args.dataset == "Covid":
            hparams['alpha'] = 0.5
            if args.test_EO == 1:
                hparams['alpha'] = 1
        elif args.dataset == "SEER":
            hparams['alpha'] = 10
    else:
        hparams['alpha'] = args.alpha
    hparams['type'] = args.type 
    hparams['lr'] = 0.01 
    hparams['mlp_width'] = 256 
    hparams['mlp_depth'] = 3
    hparams['mlp_dropout'] = 0.1
    hparams['nonlinear'] = 0
    hparams['backbone'] = "Linear" 
    
    
    ####################### use the args ###############
    hparams['n_client'] = args.n_client
    if not (args.n_step == None):
        hparams['n_step'] = args.n_step
    else:
        if args.dataset == "SEER":
            hparams['n_step'] = 500
            if hparams['agnostic_alpha'] > 0:
                hparams['n_step'] = 200
        else:
            hparams['n_step'] = 100
    if not (args.n_step == None):
        hparams['n_commu'] = args.n_commu
    else:
        if args.dataset == "SEER":
            hparams['n_commu'] = 50
            if hparams['agnostic_alpha'] > 0:
                hparams['n_commu'] = 100
        else:
            hparams['n_commu'] = 50
    ################### For AFL  ###################
    if args.dataset == "Support":
        hparams["T"] = 5
        hparams["gamma_l"] = 0.001
        hparams["gamma_w"] = 0.001
    elif args.dataset == "Covid" or args.dataset == "Cardio":
        hparams["T"] = 10
        hparams["gamma_l"] = 0.00000001
        hparams["gamma_w"] = 0.00000001
    else:
        if args.agnostic_alpha > 0:
            hparams["T"] = 5
            hparams["gamma_l"] = 0.0000001
            hparams["gamma_w"] = 0.0000001
        else:
            hparams["T"] = 2
            hparams["gamma_l"] = 0.0000001
            hparams["gamma_w"] = 0.0000001
    
    ################### For Q_FedAvg ###################s
    hparams["L_constant"] = 0.005
    hparams["q"] = 0.0005
    ################### For FairFed ####################
    hparams['beta'] = 0.1
    ###################### ours ########################
    hparams['gamma'] = 1
    hparams['A_C_alpha'] = 0.5
    hparams['rad'] = 0
    hparams['momentum_lambda'] = args.momentum_lambda
    hparams['momentum_beta'] = args.momentum_beta
    ####################################################
    set_up(hparams)
    if not (args.gamma == None):
        hparams['gamma'] = args.gamma
    if not (args.A_C_alpha == None):
        hparams['A_C_alpha'] = args.A_C_alpha
    if not (args.rad == None):
        hparams['rad'] = args.rad
    if not (args.lr == None):
        hparams['lr'] = args.lr
    #hparams['test_EO'] = 1
    
    
    hparams['sens_index'] = 0 
    if args.dataset == "Covid":
        hparams['sens_attr'] = 'Branca' # 'black' for Support ; 'Branca' for Covid
    elif args.dataset == "Support":
        hparams['sens_attr'] = 'black' # 'black' for Support ; 'Branca' for Covid

    
    return hparams
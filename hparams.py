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
from copy import deepcopy

def set_hparams(args = None):
    
    
    # for SEER
    hparams = {"alpha": 1} # for LDA
    hparams["n_client"] = 2 # for client
    hparams['holdout_fraction'] = args.holdout_fraction
    hparams['epoch_per_commu'] = args.epoch_per_commu
    hparams['gamma'] = 1
    hparams['dataset'] = args.dataset

    hparams['A_C_alpha'] = 0.5
    hparams['batch_size'] = 32 # small/large
    hparams['lr'] = 0.01 # small/larges
    hparams['mlp_width'] = 256 # small/large
    hparams['mlp_depth'] = 3
    hparams['mlp_dropout'] = 0.1
    hparams['nonlinear'] = 0
    hparams['backbone'] = "Linear" # should be set with hparams['nonlinear'] = 1
    
    ################### For AFL  ###################
    '''
    # for Support
    hparams["T"] = 5
    hparams["gamma_l"] = 0.001
    hparams["gamma_w"] = 0.001
    '''
    # For Covid and Cardio
    hparams["T"] = 10
    hparams["gamma_l"] = 0.00000001
    hparams["gamma_w"] = 0.00000001
    '''
    # for SEER agnostic
    hparams["T"] = 5
    hparams["gamma_l"] = 0.0000001
    hparams["gamma_w"] = 0.0000001
    
    # for SEER
    hparams["T"] = 2
    hparams["gamma_l"] = 0.0000001
    hparams["gamma_w"] = 0.0000001
    '''
    ################### For Q_FedAvg ###################
    
    # for others
    hparams["L_constant"] = 0.005
    hparams["q"] = 0.0005
    ################### For FairFed ####################
    hparams['beta'] = 0.1
    
    ####################### use the args ###############
    hparams['momentum_lambda'] = args.momentum_lambda
    hparams['momentum_beta'] = args.momentum_beta
    hparams['agnostic'] = 0 #use (e.g. 2019) as the agnostic
    hparams['agnostic_alpha'] = 0 # 0: not modify.
    hparams['rad'] = 0
    hparams['sens_attr'] = 'black' # 'black' for Support ; 'Branca' for Covid
    hparams['sens_index'] = 0 # only for SEER
    hparams['test_EO'] = 1
    
    return hparams
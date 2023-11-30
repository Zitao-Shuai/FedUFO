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


def print_state(args, hparams):
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

def set_up(hparams):
    dict_set = torch.load("./FedUFO/set_up.pt")
    
    if hparams['test_EO'] == 1:
        df = dict_set["test_EO"].set_index(["dataset","algorithm"])
        hparams['gamma'] = df.loc[hparams['dataset']].loc[hparams['algorithm'],"gamma"]
        hparams['A_C_alpha'] = df.loc[hparams['dataset']].loc[hparams['algorithm'],"A_C_alpha"]
        hparams['lr'] = df.loc[hparams['dataset']].loc[hparams['algorithm'],"lr"]
        hparams['rad'] = df.loc[hparams['dataset']].loc[hparams['algorithm'],"rad"]
    elif hparams['agnostic_alpha'] > 0:
        df = dict_set["agnostic_alpha"].set_index(["alpha","algorithm"])
        hparams['gamma'] = df.loc[hparams['alpha']].loc[hparams['algorithm'],"gamma"]
        hparams['A_C_alpha'] = df.loc[hparams['alpha']].loc[hparams['algorithm'],"A_C_alpha"]
        hparams['lr'] = df.loc[hparams['alpha']].loc[hparams['algorithm'],"lr"]
        hparams['rad'] = df.loc[hparams['alpha']].loc[hparams['algorithm'],"rad"]
    else:
        df = dict_set["main"].set_index(["dataset","algorithm","type"])
        hparams['gamma'] = df.loc[hparams['dataset']].loc[hparams['algorithm']].loc[hparams['type'],"gamma"]
        hparams['A_C_alpha'] = df.loc[hparams['dataset']].loc[hparams['algorithm']].loc[hparams['type'],"A_C_alpha"]
        hparams['lr'] = df.loc[hparams['dataset']].loc[hparams['algorithm']].loc[hparams['type'],"lr"]
        hparams['rad'] = df.loc[hparams['dataset']].loc[hparams['algorithm']].loc[hparams['type'],"rad"]
        
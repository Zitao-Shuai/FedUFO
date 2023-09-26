import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import torch.autograd as autograd
import collections
from torch.utils.data.dataloader import DataLoader
from copy import deepcopy
import copy
import numpy as np
from collections import OrderedDict
from FMDA.evaluate import evaluate_loss_mat
ALGORITHMS = [
    "FedAvg"
    ,"FMDA"
    ,"AFL"
    ,"Q_Fed"
    ,"Global"
    ,"Split"
    ,"FMDA_M_N"
    ,"FairFed"
    ,"DIPOLE"
    ,"FMDA_M_4"
    ,"FMDA_M_22"
    ,"FMDA_22"
    ,"FMDA_2"
    ,"FMDA_4"

]

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

class FedAvg(torch.nn.Module):
    """
    Average Federeal Training with ERM.
    """

    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        super(FedAvg, self).__init__()
        self.is_modified = 0 # to check if the client is modifier
        self.is_server = is_server # -1: server, otherwise: number of the clients
        self.num_classes = num_classes
        self.num_attr = num_classes
        self.hparams = hparams
        if hparams['backbone'] == "MLP":
            self.featurizer = MLP(input_shape, num_classes, self.hparams)
            self.classifier = Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear'])
            self.network = nn.Sequential(self.featurizer, self.classifier)
        else:
            self.featurizer = nn.Linear(input_shape, num_classes)
            self.network = self.featurizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
        )
    def evaluate_loss(self, test_data, test_model):
        step_per_epoch = int(len(test_data) / self.hparams['batch_size'])
        test_loaders = DataLoader(test_data, batch_size = self.hparams['batch_size'], shuffle=True)
        test_minibatches_iterator = iter(test_loaders)
        loss_sum = 0
        for step in range(step_per_epoch):
            X, Y = next(test_minibatches_iterator)
            Y_hat = test_model.predict(X)
            loss = F.cross_entropy(Y_hat, Y)
            loss_sum = loss_sum + loss / (step_per_epoch * self.hparams["batch_size"])
    
        return loss_sum   
    def print_state(self):
        print("modified state:", self.is_modified)
    def server_update(self, client_list, client_tr):
        # updata the server
        # input: the list of clients
        # modify the self.network
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key]
            fed_state_dict[key] = key_sum / len(client_list)
        #### update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
    
    def update(self, X, Y):
        loss = F.cross_entropy(self.predict(X), Y).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}

    def predict(self, X):
        return self.network(X)
class FairFed(torch.nn.Module):
    """
    FairFed: Enabling Group Fairness in Federated Learning
    """

    def __init__(self, input_shape, num_classes, hparams, client_tr, is_server = 0):
        super(FairFed, self).__init__()
        self.is_modified = 0 # to check if the client is modifier
        self.is_server = is_server # -1: server, otherwise: number of the clients
        self.num_classes = num_classes
        self.num_attr = num_classes
        self.hparams = hparams
        self.beta = hparams['beta']
        if hparams['backbone'] == "MLP":
            self.featurizer = MLP(input_shape, num_classes, self.hparams)
            self.classifier = Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear'])
            self.network = nn.Sequential(self.featurizer, self.classifier)
        else:
            self.featurizer = nn.Linear(input_shape, num_classes)
            self.network = self.featurizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
        ) 
        # initialize the weight vector
        self.client_number = torch.ones(hparams['n_client'])
        self.weight = torch.ones(hparams['n_client'])
        for i in range(hparams['n_client']):
            self.weight[i] = len(client_tr[i])
            self.client_number[i] = len(client_tr[i])
        self.weight = self.weight / self.weight.sum()
        from copy import deepcopy
        self.weight_use = deepcopy(self.weight)
        print("client training numbers", self.client_number)
    def print_state(self):
        print("modified state:", self.is_modified)
    def server_update(self, client_list, client_tr, EO_list, ACC_list):
        # updata the server
        # input: the list of clients
        # modify the self.network
        F_global = 0
        acc_global = 0
        for i in range(len(EO_list)):
            F_global = self.client_number[i] * EO_list[i]/self.client_number.sum()
            acc_global = self.client_number[i] * ACC_list[i]/self.client_number.sum()
        delta_vector = torch.ones(self.hparams['n_client'])
        for i in range(len(EO_list)):
            delta_vector[i] = (F_global - EO_list[i])
            if delta_vector[i] < 0:
                delta_vector[i] = -delta_vector[i]
        delta_global = delta_vector.mean()
        for i in range(len(EO_list)):
            self.weight[i] = self.weight[i] - self.beta * (delta_vector[i] - delta_global)
        for i in range(len(EO_list)):
            self.weight_use[i] = self.weight[i]/self.weight.sum()
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key] * self.weight_use[i]
            fed_state_dict[key] = key_sum / len(client_list)
        #### update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
    
    def update(self, X, Y):
        
        loss = F.cross_entropy(self.predict(X), Y).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}

    def predict(self, X):
        return self.network(X)
    
class DIPOLE(torch.nn.Module):
    """
    Improving Fairness in AI Models on Electronic Health Records: The Case for Federated Learning Methods
    """

    def __init__(self, input_shape, num_classes, hparams, client_tr, is_server = 0):
        super(DIPOLE, self).__init__()
        self.is_modified = 0 # to check if the client is modifier
        self.is_server = is_server # -1: server, otherwise: number of the clients
        self.num_classes = num_classes
        self.num_attr = num_classes
        self.hparams = hparams
        self.beta = hparams['beta']
        if hparams['backbone'] == "MLP":
            self.featurizer = MLP(input_shape, num_classes, self.hparams)
            self.classifier = Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear'])
            self.network = nn.Sequential(self.featurizer, self.classifier)
        else:
            self.featurizer = nn.Linear(input_shape, num_classes)
            self.network = self.featurizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
        ) 
        # initialize the weight vector
        self.client_number = torch.ones(hparams['n_client'])
        self.weight = torch.ones(hparams['n_client'])
        for i in range(hparams['n_client']):
            self.weight[i] = len(client_tr[i])
            self.client_number[i] = len(client_tr[i])
        self.weight = self.weight / self.weight.sum()
        from copy import deepcopy
        self.weight_use = deepcopy(self.weight)
        print("client training numbers", self.client_number)
    def print_state(self):
        print("modified state:", self.is_modified)
    def server_update(self, client_list, client_tr, EO_list, ACC_list):
        # updata the server
        # input: the list of clients
        # modify the self.network
        F_global = 0
        acc_global = 0
        EO_max = 0
        for i in range(len(EO_list)):
            F_global = self.client_number[i] * EO_list[i]/self.client_number.sum()
            acc_global = self.client_number[i] * ACC_list[i]/self.client_number.sum()
            if EO_max < EO_list[i]:
                EO_max = EO_list[i]
        
        for i in range(len(EO_list)):
            self.weight[i] = self.weight[i] + self.beta * (EO_max - EO_list[i])

        for i in range(len(EO_list)):
            self.weight_use[i] = self.weight[i]/self.weight.sum()
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key] * self.weight_use[i]
            fed_state_dict[key] = key_sum / len(client_list)
        #### update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
    
    def update(self, X, Y):
        
        loss = F.cross_entropy(self.predict(X), Y).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}

    def predict(self, X):
        return self.network(X)
    
class Global(FedAvg):
    """
    Global model.
    """
    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        # can only be used in the case that there is only 1 client
        super(Global, self).__init__(input_shape, num_classes, hparams, is_server)
        self.w = torch.ones(hparams['n_client']) * 1.0 / hparams['n_client']
        self.lambda_mat = torch.ones((hparams['n_client'],num_classes)) * 1.0 / (hparams['n_client']**2) 
    def server_update(self, client_list, client_tr):
        if len(client_list) > 1:
            print("ERROR! The number of clients should not bigger than 1!")
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key]
            fed_state_dict[key] = key_sum / len(client_list)
        #### update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
class Split(FedAvg):
    """
    Splitly training model. The server does nothing.
    """
    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        # can only be used in the case that there is only 1 client
        super(Split, self).__init__(input_shape, num_classes, hparams, is_server)
        self.w = torch.ones(hparams['n_client']) * 1.0 / hparams['n_client']
        self.lambda_mat = torch.ones((hparams['n_client'],num_classes)) * 1.0 / (hparams['n_client']**2) 
    def server_update(self, client_list, client_tr):
        pass

def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and (v >= 0).all():
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.flip(torch.sort(v)[0],dims=(0,))
    cssv = torch.cumsum(u,dim=0)
    # get the number of > 0 components of the optimal solution
    non_zero_vector = torch.nonzero(u * torch.arange(1, n+1) > (cssv - s), as_tuple=False)
    if len(non_zero_vector) == 0:
        rho=0.0
    else:
        rho = non_zero_vector[-1].squeeze()
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w
def project_onto_chi_square_ball(w, rho, tol = 1e-10):
    assert (rho > 0)
    rho = float(rho)
  
    # sort in decreasing order
    w_sort = np.sort(w) # increasing
    w_sort = w_sort[::-1] # decreasing
  
    w_sort_cumsum = w_sort.cumsum()
    w_sort_sqr_cumsum = np.square(w_sort).cumsum()
    nn = float(w_sort.shape[0])

    lam_min = 0.0
    lam_max = (1/nn) * (nn * w_sort[0] / np.sqrt(2. * rho + 1.) - 1.)
    lam_init_max = lam_max

    if (lam_max <= 0): # optimal lambda is 0
        (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, 0., rho)
        p = w - eta
        low_inds = p < 0
        p[low_inds] = 0.
        return p

    # bisect on lambda to find the optimal lambda value
    while (lam_max - lam_min > tol * lam_init_max):
        lam = .5 * (lam_max + lam_min)
        (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho)

        # compute norm(p(lam))^2 * (1+lam * nn)^2
        thresh = .5 * nn * (w_sort_sqr_cumsum[ind] - 2. * eta * w_sort_cumsum[ind] + eta**2 * (ind+1.))
        if (thresh > (rho + .5) * (1 + lam * nn)**2):
            # constraint infeasible, increase lam (dual var)
            lam_min = lam
        else:
            # constraint loose, decrease lam (dual var)
            lam_max = lam

    lam = .5 * (lam_max + lam_min)
    (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho)
    p = w - eta
    low_inds = p < 0
    p[low_inds] = 0
    return (1. / (1. + lam * nn)) * p
def solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho):
    fs = w_sort - (w_sort_cumsum - (1. + lam * nn)) / (np.arange(nn) + 1.)
    ind = (fs > 0).sum()-1
    return ((1 / (ind+1.)) * (w_sort_cumsum[ind] - (1. + lam * nn)), ind)



        
class FMDA(FedAvg):
    """
    Our method with ERM.
    For the N * M case.
    """

    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        super(FMDA, self).__init__(input_shape, num_classes, hparams, is_server)
        self.n_client = hparams['n_client']
        self.lambda_mat = torch.ones((hparams['n_client'],num_classes)) * 1.0 / (hparams['n_client'] * num_classes) 
        self.momentum_beta = hparams['momentum_beta']
        self.momentum_lambda = hparams['momentum_lambda']
    def evaluate_loss_self(self, test_data, hparams):
        step_per_epoch = int(len(test_data) / hparams['batch_size'])
        test_loaders = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle=True)
        test_minibatches_iterator = iter(test_loaders)
        loss_sum = 0
        for step in range(step_per_epoch):
            X, Y = next(test_minibatches_iterator)
            Y_hat = self.predict(X)
            loss = F.cross_entropy(Y_hat, Y)
            loss_sum = loss_sum + loss / (step_per_epoch)
    
        return loss_sum   
    def evaluate_loss_mat(self, te_dataset, hparams):
        # evaluate the performance of the client models
        loss_mat = torch.zeros((len(te_dataset),self.num_attr))
        for id_te, te in enumerate (te_dataset):#id_te == id_client
            attr_list = te.split_by_attr()
            for id_attr, attr in enumerate(attr_list):
                loss = self.evaluate_loss_self(attr, hparams)
                loss_mat[id_te, id_attr] = loss
        return loss_mat
    def server_update(self, client_list, client_tr):
        # updata the server
        # input: the list of clients
        # modify the self.network using our method

        # step 1: get the parameters
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        old_state_dict = self.network.state_dict() # for momentum update
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()

        # step 2: updata the model using the weight vector

        for index, key in enumerate(weight_keys):
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key] * 1 / self.n_client
            fed_state_dict[key] = key_sum * (1 + self.momentum_beta) - self.momentum_beta * old_state_dict[key]
        # step 3: update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
        # step 4: get the loss on training set of clients and update the w vector
        # matrix: n_client * n_dataset
        loss_mat = self.evaluate_loss_mat(client_tr, self.hparams)
        E = self.hparams['n_step']/25
        gamma = self.hparams['gamma']
        from copy import deepcopy
        old_lambda_mat = self.lambda_mat * 1.0 # for momemtum update
        lambda_mat = (self.lambda_mat).log() + (E * gamma * loss_mat)
        lambda_mat = (lambda_mat - (lambda_mat.exp().sum()).log()).exp()
        self.lambda_mat = lambda_mat * self.momentum_lambda + (1 - self.momentum_lambda) * old_lambda_mat # will be sent to server
        self.lambda_mat = self.lambda_mat / self.lambda_mat.sum() #normalization
        
        # for the rad
        if self.hparams['rad'] > 0:
            for i in range(self.lambda_mat.shape[0]):
                self.lambda_mat[i, :] =  project_onto_chi_square_ball(self.lambda_mat[i, :].detach(), self.hparams['rad'])
        return self.lambda_mat.detach()
    def set_lambda_mat(self, lambda_mat):
        self.lambda_mat = lambda_mat
    def update(self, X, Y):
        # update based on the number
        w = self.lambda_mat[self.is_server, :]
        
        loss = F.cross_entropy(self.predict(X), Y, weight = w).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
class FMDA_2(FedAvg):
    """
    Our method with ERM.
    For the N * M case.
    # Only for SEER.
    """

    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        super(FMDA_2, self).__init__(input_shape, num_classes, hparams, is_server)
        
        self.n_client = hparams['n_client']
        self.num_attr = 2
        self.num_classes = num_classes
        self.lambda_mat = torch.ones((hparams['n_client'],2)) * 1.0 / (hparams['n_client'] * 2) # Sensitive: 2
        self.momentum_beta = hparams['momentum_beta']
        self.momentum_lambda = hparams['momentum_lambda']
    def evaluate_loss_self(self, test_data, hparams):
        step_per_epoch = int(len(test_data) / hparams['batch_size'])
        test_loaders = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle=True)
        test_minibatches_iterator = iter(test_loaders)
        loss_sum = 0
        for step in range(step_per_epoch):
            X, Y = next(test_minibatches_iterator)
            Y_hat = self.predict(X)
            loss = F.cross_entropy(Y_hat, Y)
            loss_sum = loss_sum + loss / (step_per_epoch)
    
        return loss_sum   
    def evaluate_loss_mat(self, te_dataset, hparams):
        # evaluate the performance of the client models
        loss_mat = torch.zeros((len(te_dataset),self.num_attr))
        for id_te, te in enumerate (te_dataset):#id_te == id_client
            attr_list = te.split_by_attr_2(self.num_attr, self.hparams['sens_index'])
            for id_attr, attr in enumerate(attr_list):
                loss = self.evaluate_loss_self(attr, hparams)
                loss_mat[id_te, id_attr] = loss
        return loss_mat
    def server_update(self, client_list, client_tr):
        # updata the server
        # input: the list of clients
        # modify the self.network using our method

        # step 1: get the parameters
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        old_state_dict = self.network.state_dict() # for momentum update
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()

        # step 2: updata the model using the weight vector

        for index, key in enumerate(weight_keys):
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key] * 1 / self.n_client
            fed_state_dict[key] = key_sum * (1 + self.momentum_beta) - self.momentum_beta * old_state_dict[key]
        # step 3: update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
        # step 4: get the loss on training set of clients and update the w vector
        # matrix: n_client * n_dataset
        loss_mat = self.evaluate_loss_mat(client_tr, self.hparams)
        E = self.hparams['n_step']/25
        gamma = self.hparams['gamma']
        from copy import deepcopy
        old_lambda_mat = self.lambda_mat * 1.0 # for momemtum update
        lambda_mat = (self.lambda_mat).log() + (E * gamma * loss_mat)
        lambda_mat = (lambda_mat - (lambda_mat.exp().sum()).log()).exp()
        self.lambda_mat = lambda_mat * self.momentum_lambda + (1 - self.momentum_lambda) * old_lambda_mat # will be sent to server
        self.lambda_mat = self.lambda_mat / self.lambda_mat.sum() #normalization
        
        # for the rad
        if self.hparams['rad'] > 0:
            for i in range(self.lambda_mat.shape[0]):
                self.lambda_mat[i, :] =  project_onto_chi_square_ball(self.lambda_mat[i, :].detach(), self.hparams['rad'])
        return self.lambda_mat.detach()
    def set_lambda_mat(self, lambda_mat):
        self.lambda_mat = lambda_mat
    def update(self, X, Y):
        # update based on the number
        w = self.lambda_mat[self.is_server, :]
        sen = X[:, self.hparams['sens_index']]
        
        w_temp = torch.zeros(X.shape[0]).to("cpu")
        for i in range(X.shape[0]):
            if sen[i] == 1:
                w_temp[i] = w[1]
            else:
                w_temp[i] = w[0]
            
        loss = (F.cross_entropy(self.predict(X), Y, reduce = False) * w_temp).mean()
            
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
class FMDA_22(FedAvg):
    """
    Our method with ERM.
    For the N * M case.
    # Only for SEER.
    """

    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        super(FMDA_22, self).__init__(input_shape, num_classes, hparams, is_server)
        self.n_client = hparams['n_client']
        self.num_attr = 2 * num_classes
        self.num_classes = num_classes
        self.lambda_mat = torch.ones((hparams['n_client'],2 * num_classes)) * 1.0 / (hparams['n_client'] * (2 * num_classes)) # sensitive: 2 
        self.momentum_beta = hparams['momentum_beta']
        self.momentum_lambda = hparams['momentum_lambda']
    def evaluate_loss_self(self, test_data, hparams):
        step_per_epoch = int(len(test_data) / hparams['batch_size'])

        if len(test_data) > 0:
            test_loaders = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle=True)
            test_minibatches_iterator = iter(test_loaders)
            loss_sum = 0
            for step in range(step_per_epoch):
                X, Y = next(test_minibatches_iterator)
                Y_hat = self.predict(X)
                loss = F.cross_entropy(Y_hat, Y)
                loss_sum = loss_sum + loss / (step_per_epoch)
    
            return loss_sum   
        else:
            # this combination is null return a zero
            return 0
    def evaluate_loss_mat(self, te_dataset, hparams):
        # evaluate the performance of the client models
        loss_mat = torch.zeros((len(te_dataset),self.num_attr))
        for id_te, te in enumerate (te_dataset):#id_te == id_client
            attr_list = te.split_by_attr_22(self.num_attr, self.hparams['sens_index'])
            for id_attr, attr in enumerate(attr_list):
                loss = self.evaluate_loss_self(attr, hparams)
                loss_mat[id_te, id_attr] = loss
        return loss_mat
    def server_update(self, client_list, client_tr):
        # updata the server
        # input: the list of clients
        # modify the self.network using our method

        # step 1: get the parameters
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        old_state_dict = self.network.state_dict() # for momentum update
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()

        # step 2: updata the model using the weight vector

        for index, key in enumerate(weight_keys):
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key] * 1 / self.n_client
            fed_state_dict[key] = key_sum * (1 + self.momentum_beta) - self.momentum_beta * old_state_dict[key]
        # step 3: update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
        # step 4: get the loss on training set of clients and update the w vector
        # matrix: n_client * n_dataset
        loss_mat = self.evaluate_loss_mat(client_tr, self.hparams)
        E = self.hparams['n_step']/25
        gamma = self.hparams['gamma']
        from copy import deepcopy
        old_lambda_mat = self.lambda_mat * 1.0 # for momemtum update
        lambda_mat = (self.lambda_mat).log() + (E * gamma * loss_mat)
        lambda_mat = (lambda_mat - (lambda_mat.exp().sum()).log()).exp()
        self.lambda_mat = lambda_mat * self.momentum_lambda + (1 - self.momentum_lambda) * old_lambda_mat # will be sent to server
        self.lambda_mat = self.lambda_mat / self.lambda_mat.sum() #normalization
        
        # for the rad
        if self.hparams['rad'] > 0:
            for i in range(self.lambda_mat.shape[0]):
                self.lambda_mat[i, :] =  project_onto_chi_square_ball(self.lambda_mat[i, :].detach(), self.hparams['rad'])
        return self.lambda_mat.detach()
    def set_lambda_mat(self, lambda_mat):
        self.lambda_mat = lambda_mat
    def update(self, X, Y):
        # update based on the number
        # sen 1: 1 * 11 + label_number 
        # sen 0: 0 * 11 + label_number
        w = self.lambda_mat[self.is_server, :]
        sen = X[:, self.hparams['sens_index']]
        
        
        w_temp = torch.zeros(X.shape[0]).to("cpu")
        for i in range(X.shape[0]):
            if sen[i] == 1:
                w_temp[i] = w[self.num_classes + Y[i]]
            else:
                w_temp[i] = w[Y[i]]
            
        loss = (F.cross_entropy(self.predict(X), Y, reduce = False) * w_temp).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
class FMDA_4(FedAvg):
    """
    Our method with ERM.
    For the N * M case.
    # Only for SEER.
    """

    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        super(FMDA_4, self).__init__(input_shape, num_classes, hparams, is_server)
        self.n_client = hparams['n_client']
        self.num_attr = 2 * num_classes
        self.num_classes = num_classes
        self.lambda_mat = torch.ones((hparams['n_client'],2 * num_classes)) * 1.0 / (hparams['n_client'] * (2 * num_classes)) # sensitive: 2 
        self.momentum_beta = hparams['momentum_beta']
        self.momentum_lambda = hparams['momentum_lambda']
    def evaluate_loss_self(self, test_data, hparams):
        step_per_epoch = int(len(test_data) / hparams['batch_size'])
        #print("len(test_data):", len(test_data))
        if len(test_data) > 0:
            test_loaders = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle=True)
            test_minibatches_iterator = iter(test_loaders)
            loss_sum = 0
            for step in range(step_per_epoch):
                X, Y = next(test_minibatches_iterator)
                Y_hat = self.predict(X)
                loss = F.cross_entropy(Y_hat, Y)
                loss_sum = loss_sum + loss / (step_per_epoch)
    
            return loss_sum   
        else:
            # this combination is null return a zero
            return 0
    def evaluate_loss_mat(self, te_dataset, hparams):
        # evaluate the performance of the client models
        loss_mat = torch.zeros((len(te_dataset),self.num_attr))
        for id_te, te in enumerate (te_dataset):#id_te == id_client
            attr_list = te.split_by_attr_4(self.num_attr, self.hparams['sens_index'])
            for id_attr, attr in enumerate(attr_list):
                loss = self.evaluate_loss_self(attr, hparams)
                loss_mat[id_te, id_attr] = loss
        return loss_mat
    def server_update(self, client_list, client_tr):
        # updata the server
        # input: the list of clients
        # modify the self.network using our method

        # step 1: get the parameters
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        old_state_dict = self.network.state_dict() # for momentum update
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()

        # step 2: updata the model using the weight vector

        for index, key in enumerate(weight_keys):
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key] * 1 / self.n_client
            fed_state_dict[key] = key_sum * (1 + self.momentum_beta) - self.momentum_beta * old_state_dict[key]
        # step 3: update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
        # step 4: get the loss on training set of clients and update the w vector
        # matrix: n_client * n_dataset
        loss_mat = self.evaluate_loss_mat(client_tr, self.hparams)
        E = self.hparams['n_step']/25
        gamma = self.hparams['gamma']
        from copy import deepcopy
        old_lambda_mat = self.lambda_mat * 1.0 # for momemtum update
        lambda_mat = (self.lambda_mat).log() + (E * gamma * loss_mat)
        lambda_mat = (lambda_mat - (lambda_mat.exp().sum()).log()).exp()
        self.lambda_mat = lambda_mat * self.momentum_lambda + (1 - self.momentum_lambda) * old_lambda_mat # will be sent to server
        self.lambda_mat = self.lambda_mat / self.lambda_mat.sum() #normalization
        
        # for the rad
        if self.hparams['rad'] > 0:
            for i in range(self.lambda_mat.shape[0]):
                self.lambda_mat[i, :] =  project_onto_chi_square_ball(self.lambda_mat[i, :].detach(), self.hparams['rad'])
        return self.lambda_mat.detach()
    def set_lambda_mat(self, lambda_mat):
        self.lambda_mat = lambda_mat
    def update(self, X, Y):
        # update based on the number
        # sen 1: 1 * 2 + label_number 
        # sen 0: 0 * 2 + label_number
        w = self.lambda_mat[self.is_server, :]
        sen = X[:, self.hparams['sens_index']]
        
        
        w_temp = torch.zeros(X.shape[0]).to("cpu")
        for i in range(X.shape[0]):
            if sen[i] == 1:
                w_temp[i] = w[self.num_classes + Y[i]]
            else:
                w_temp[i] = w[Y[i]]
            
        loss = (F.cross_entropy(self.predict(X), Y, reduce = False) * w_temp).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
class FMDA_M_4(FedAvg):
    """
    Our method with ERM.
    For the M + N case.
    """

    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        super(FMDA_M_4, self).__init__(input_shape, num_classes, hparams, is_server)
        self.A_C_alpha = self.hparams['A_C_alpha']
        self.n_client = hparams['n_client']
        self.num_attr = 2 * num_classes
        self.lambda_mat_A = torch.ones(num_classes * 2) * 1.0 / (num_classes * 2) 
        self.lambda_mat_C = torch.ones(hparams['n_client']) * 1.0 / (hparams['n_client']) #no use, hence no modification
        self.momentum_beta = hparams['momentum_beta']
        self.momentum_lambda = hparams['momentum_lambda']
    def evaluate_loss_self(self, test_data, hparams):
        step_per_epoch = int(len(test_data) / hparams['batch_size'])
        test_loaders = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle=True)
        test_minibatches_iterator = iter(test_loaders)
        loss_sum = 0
        for step in range(step_per_epoch):
            X, Y = next(test_minibatches_iterator)
            Y_hat = self.predict(X)
            loss = F.cross_entropy(Y_hat, Y)
            loss_sum = loss_sum + loss / (step_per_epoch)
    
        return loss_sum   
    def evaluate_loss_mat(self, te_dataset, hparams):
        # evaluate the performance of the client models
        loss_mat = torch.zeros((len(te_dataset),self.num_attr))
        number_mat = torch.zeros((len(te_dataset),self.num_attr))
        for id_te, te in enumerate (te_dataset):#id_te == id_client
            attr_list, number_vector = te.split_by_attr_4(self.num_attr, self.hparams['sens_index'])
            for id_attr, attr in enumerate(attr_list):
                loss = self.evaluate_loss_self(attr, hparams)
                loss_mat[id_te, id_attr] = loss
                number_mat[id_te, id_attr] = number_vector[id_attr]
        return loss_mat, number_mat
    def server_update(self, client_list, client_tr):
        # updata the server
        # input: the list of clients
        # modify the self.network using our method

        # step 1: get the parameters
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        old_state_dict = self.network.state_dict() # for momentum update
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()

        # step 2: updata the model using the weight vector

        for index, key in enumerate(weight_keys):
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key] * 1 / self.n_client
            fed_state_dict[key] = key_sum * (1 + self.momentum_beta) - self.momentum_beta * old_state_dict[key]
        # step 3: update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
        # step 4: get the loss on training set of clients and update the w vector
        # matrix: n_client * n_dataset
        loss_mat, number_mat = self.evaluate_loss_mat(client_tr, self.hparams)
        E = self.hparams['n_step']/25
        gamma = self.hparams['gamma']
        
        
        from copy import deepcopy
        old_lambda_mat_A = self.lambda_mat_A * 1.0 # for momemtum update
        loss_vector_A = torch.ones(loss_mat.shape[1]) * 1.0
        for i in range(loss_mat.shape[1]):
            temp_sum = 0
            for j in range(loss_mat.shape[0]):
                temp_sum += loss_mat[j,i] * number_mat[j,i]/number_mat[:,i].sum()
            loss_vector_A[i] = temp_sum
        lambda_mat_A = (self.lambda_mat_A).log() + (E * gamma * loss_vector_A)
        
        
        
        lambda_mat_A = (lambda_mat_A - (lambda_mat_A.exp().sum()).log()).exp()
        self.lambda_mat_A = lambda_mat_A * self.momentum_lambda + (1 - self.momentum_lambda) * old_lambda_mat_A # will be sent to server
        self.lambda_mat_A = self.lambda_mat_A / self.lambda_mat_A.sum() #normalization
        
        old_lambda_mat_C = self.lambda_mat_C * 1.0 # for momemtum update
        loss_vector_C = torch.ones(self.hparams['n_client']) * 1.0
        for i in range(loss_mat.shape[0]):
            temp_sum = 0
            for j in range(loss_mat.shape[1]):
                temp_sum += loss_mat[i,j] * number_mat[i,j]/number_mat[i,:].sum()
            loss_vector_C[i] = temp_sum
        lambda_mat_C = (self.lambda_mat_C).log() + (E * gamma * loss_vector_C)
        
        
        
        lambda_mat_C = (lambda_mat_C - (lambda_mat_C.exp().sum()).log()).exp()
        self.lambda_mat_C = lambda_mat_C * self.momentum_lambda + (1 - self.momentum_lambda) * old_lambda_mat_C # will be sent to server
        self.lambda_mat_C = self.lambda_mat_C / self.lambda_mat_C.sum() #normalization

        # for the rad
        if self.hparams['rad'] > 0:
            print("server update", self.lambda_mat_A)
            self.lambda_mat_A =  project_onto_chi_square_ball(self.lambda_mat_A.detach(), self.hparams['rad'])
            self.lambda_mat_C =  project_onto_chi_square_ball(self.lambda_mat_C.detach(), self.hparams['rad'])
            
        self.lambda_mat_A = self.lambda_mat_A / self.lambda_mat_A.sum() #normalization
        self.lambda_mat_C = self.lambda_mat_C / self.lambda_mat_C.sum() #normalization
        
        return self.lambda_mat_A.detach(), self.lambda_mat_C.detach()
    
    
    
    def set_lambda_mat(self, lambda_mat_A, lambda_mat_C):
        self.lambda_mat_A = lambda_mat_A
        self.lambda_mat_C = lambda_mat_C
    def update(self, X, Y):
        # update based on the number
        # sen 1: 1 * 2 + label_number 
        # sen 0: 0 * 2 + label_number
        w = self.lambda_mat_A
        sen = X[:, self.hparams['sens_index']]
        
        w_temp = torch.zeros(X.shape[0]).to("cpu")
        for i in range(X.shape[0]):
            if sen[i] == 1:
                w_temp[i] = w[self.num_classes + Y[i]]
            else:
                w_temp[i] = w[Y[i]]
            
        loss = (F.cross_entropy(self.predict(X), Y, reduce = False) * w_temp).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
class FMDA_M_22(FedAvg):
    """
    Our method with ERM.
    For the M + N case.
    """

    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        super(FMDA_M_4, self).__init__(input_shape, num_classes, hparams, is_server)
        self.A_C_alpha = self.hparams['A_C_alpha']
        self.n_client = hparams['n_client']
        self.num_attr = 2 * num_classes
        self.lambda_mat_A = torch.ones(num_classes * 11) * 1.0 / (num_classes * 11) 
        self.lambda_mat_C = torch.ones(hparams['n_client']) * 1.0 / (hparams['n_client']) #no use, hence no modification
        self.momentum_beta = hparams['momentum_beta']
        self.momentum_lambda = hparams['momentum_lambda']
    def evaluate_loss_self(self, test_data, hparams):
        step_per_epoch = int(len(test_data) / hparams['batch_size'])
        test_loaders = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle=True)
        test_minibatches_iterator = iter(test_loaders)
        loss_sum = 0
        for step in range(step_per_epoch):
            X, Y = next(test_minibatches_iterator)
            Y_hat = self.predict(X)
            loss = F.cross_entropy(Y_hat, Y)
            loss_sum = loss_sum + loss / (step_per_epoch)
    
        return loss_sum   
    def evaluate_loss_mat(self, te_dataset, hparams):
        # evaluate the performance of the client models
        loss_mat = torch.zeros((len(te_dataset),self.num_attr))
        number_mat = torch.zeros((len(te_dataset),self.num_attr))
        for id_te, te in enumerate (te_dataset):#id_te == id_client
            attr_list, number_vector = te.split_by_attr_22(self.num_attr, self.hparams['sens_index'])
            for id_attr, attr in enumerate(attr_list):
                loss = self.evaluate_loss_self(attr, hparams)
                loss_mat[id_te, id_attr] = loss
                number_mat[id_te, id_attr] = number_vector[id_attr]
        return loss_mat, number_mat
    def server_update(self, client_list, client_tr):
        # updata the server
        # input: the list of clients
        # modify the self.network using our method

        # step 1: get the parameters
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        old_state_dict = self.network.state_dict() # for momentum update
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()

        # step 2: updata the model using the weight vector

        for index, key in enumerate(weight_keys):
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key] * 1 / self.n_client
            fed_state_dict[key] = key_sum * (1 + self.momentum_beta) - self.momentum_beta * old_state_dict[key]
        # step 3: update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
        # step 4: get the loss on training set of clients and update the w vector
        # matrix: n_client * n_dataset
        loss_mat, number_mat = self.evaluate_loss_mat(client_tr, self.hparams)
        E = self.hparams['n_step']/25
        gamma = self.hparams['gamma']
        
        
        from copy import deepcopy
        old_lambda_mat_A = self.lambda_mat_A * 1.0 # for momemtum update
        loss_vector_A = torch.ones(loss_mat.shape[1]) * 1.0
        for i in range(loss_mat.shape[1]):
            temp_sum = 0
            for j in range(loss_mat.shape[0]):
                temp_sum += loss_mat[j,i] * number_mat[j,i]/number_mat[:,i].sum()
            loss_vector_A[i] = temp_sum
        lambda_mat_A = (self.lambda_mat_A).log() + (E * gamma * loss_vector_A)
        
        
        
        lambda_mat_A = (lambda_mat_A - (lambda_mat_A.exp().sum()).log()).exp()
        self.lambda_mat_A = lambda_mat_A * self.momentum_lambda + (1 - self.momentum_lambda) * old_lambda_mat_A # will be sent to server
        self.lambda_mat_A = self.lambda_mat_A / self.lambda_mat_A.sum() #normalization
        
        old_lambda_mat_C = self.lambda_mat_C * 1.0 # for momemtum update
        loss_vector_C = torch.ones(self.hparams['n_client']) * 1.0
        for i in range(loss_mat.shape[0]):
            temp_sum = 0
            for j in range(loss_mat.shape[1]):
                temp_sum += loss_mat[i,j] * number_mat[i,j]/number_mat[i,:].sum()
            loss_vector_C[i] = temp_sum
        lambda_mat_C = (self.lambda_mat_C).log() + (E * gamma * loss_vector_C)
        
        
        
        lambda_mat_C = (lambda_mat_C - (lambda_mat_C.exp().sum()).log()).exp()
        self.lambda_mat_C = lambda_mat_C * self.momentum_lambda + (1 - self.momentum_lambda) * old_lambda_mat_C # will be sent to server
        self.lambda_mat_C = self.lambda_mat_C / self.lambda_mat_C.sum() #normalization

        # for the rad
        if self.hparams['rad'] > 0:
            print("server update", self.lambda_mat_A)
            self.lambda_mat_A =  project_onto_chi_square_ball(self.lambda_mat_A.detach(), self.hparams['rad'])
            self.lambda_mat_C =  project_onto_chi_square_ball(self.lambda_mat_C.detach(), self.hparams['rad'])
            
        self.lambda_mat_A = self.lambda_mat_A / self.lambda_mat_A.sum() #normalization
        self.lambda_mat_C = self.lambda_mat_C / self.lambda_mat_C.sum() #normalization
        
        return self.lambda_mat_A.detach(), self.lambda_mat_C.detach()
    
    
    
    def set_lambda_mat(self, lambda_mat_A, lambda_mat_C):
        self.lambda_mat_A = lambda_mat_A
        self.lambda_mat_C = lambda_mat_C
    def update(self, X, Y):
        # update based on the number
        # sen 1: 1 * 2 + label_number 
        # sen 0: 0 * 2 + label_number
        w = self.lambda_mat_A
        sen = X[:, self.hparams['sens_index']]
        
        w_temp = torch.zeros(X.shape[0]).to("cpu")
        for i in range(X.shape[0]):
            if sen[i] == 1:
                w_temp[i] = w[self.num_classes + Y[i]]
            else:
                w_temp[i] = w[Y[i]]
            
        loss = (F.cross_entropy(self.predict(X), Y, reduce = False) * w_temp).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
class FMDA_M_N(FedAvg):
    """
    Our method with ERM.
    For the M + N case.
    """

    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        super(FMDA_M_N, self).__init__(input_shape, num_classes, hparams, is_server)
        self.A_C_alpha = self.hparams['A_C_alpha']
        self.n_client = hparams['n_client']
        self.lambda_mat_A = torch.ones(num_classes) * 1.0 / (num_classes) 
        self.lambda_mat_C = torch.ones(hparams['n_client']) * 1.0 / (hparams['n_client']) 
        self.momentum_beta = hparams['momentum_beta']
        self.momentum_lambda = hparams['momentum_lambda']
    def evaluate_loss_self(self, test_data, hparams):
        step_per_epoch = int(len(test_data) / hparams['batch_size'])
        test_loaders = DataLoader(test_data, batch_size = hparams['batch_size'], shuffle=True)
        test_minibatches_iterator = iter(test_loaders)
        loss_sum = 0
        for step in range(step_per_epoch):
            X, Y = next(test_minibatches_iterator)
            Y_hat = self.predict(X)
            loss = F.cross_entropy(Y_hat, Y)
            loss_sum = loss_sum + loss / (step_per_epoch)
    
        return loss_sum   
    def evaluate_loss_mat(self, te_dataset, hparams):
        # evaluate the performance of the client models
        loss_mat = torch.zeros((len(te_dataset),self.num_attr))
        number_mat = torch.zeros((len(te_dataset),self.num_attr))
        for id_te, te in enumerate (te_dataset):#id_te == id_client
            attr_list, number_vector = te.split_by_attr_overall()
            for id_attr, attr in enumerate(attr_list):
                loss = self.evaluate_loss_self(attr, hparams)
                loss_mat[id_te, id_attr] = loss
                number_mat[id_te, id_attr] = number_vector[id_attr]
        return loss_mat, number_mat
    def server_update(self, client_list, client_tr):
        # updata the server
        # input: the list of clients
        # modify the self.network using our method

        # step 1: get the parameters
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        old_state_dict = self.network.state_dict() # for momentum update
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        #print(self.w)
        # step 2: updata the model using the weight vector

        for index, key in enumerate(weight_keys):
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key] * 1 / self.n_client
            fed_state_dict[key] = key_sum * (1 + self.momentum_beta) - self.momentum_beta * old_state_dict[key]
        # step 3: update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
        # step 4: get the loss on training set of clients and update the w vector
        # matrix: n_client * n_dataset
        loss_mat, number_mat = self.evaluate_loss_mat(client_tr, self.hparams)
        E = self.hparams['n_step']/25
        gamma = self.hparams['gamma']
        
        
        from copy import deepcopy
        old_lambda_mat_A = self.lambda_mat_A * 1.0 # for momemtum update
        loss_vector_A = torch.ones(loss_mat.shape[1]) * 1.0
        for i in range(loss_mat.shape[1]):
            temp_sum = 0
            for j in range(loss_mat.shape[0]):
                temp_sum += loss_mat[j,i] * number_mat[j,i]/number_mat[:,i].sum()
            loss_vector_A[i] = temp_sum
        lambda_mat_A = (self.lambda_mat_A).log() + (E * gamma * loss_vector_A)
        
        
        
        lambda_mat_A = (lambda_mat_A - (lambda_mat_A.exp().sum()).log()).exp()
        self.lambda_mat_A = lambda_mat_A * self.momentum_lambda + (1 - self.momentum_lambda) * old_lambda_mat_A # will be sent to server
        self.lambda_mat_A = self.lambda_mat_A / self.lambda_mat_A.sum() #normalization
        
        old_lambda_mat_C = self.lambda_mat_C * 1.0 # for momemtum update
        loss_vector_C = torch.ones(self.hparams['n_client']) * 1.0
        for i in range(loss_mat.shape[0]):
            temp_sum = 0
            for j in range(loss_mat.shape[1]):
                temp_sum += loss_mat[i,j] * number_mat[i,j]/number_mat[i,:].sum()
            loss_vector_C[i] = temp_sum
        lambda_mat_C = (self.lambda_mat_C).log() + (E * gamma * loss_vector_C)
        
        
        
        lambda_mat_C = (lambda_mat_C - (lambda_mat_C.exp().sum()).log()).exp()
        self.lambda_mat_C = lambda_mat_C * self.momentum_lambda + (1 - self.momentum_lambda) * old_lambda_mat_C # will be sent to server
        self.lambda_mat_C = self.lambda_mat_C / self.lambda_mat_C.sum() #normalization

        # for the rad
        if self.hparams['rad'] > 0:
            print("server update", self.lambda_mat_A)
            self.lambda_mat_A =  project_onto_chi_square_ball(self.lambda_mat_A.detach(), self.hparams['rad'])
            self.lambda_mat_C =  project_onto_chi_square_ball(self.lambda_mat_C.detach(), self.hparams['rad'])
            
        self.lambda_mat_A = self.lambda_mat_A / self.lambda_mat_A.sum() #normalization
        self.lambda_mat_C = self.lambda_mat_C / self.lambda_mat_C.sum() #normalization
        
        return self.lambda_mat_A.detach(), self.lambda_mat_C.detach()
    
    
    
    def set_lambda_mat(self, lambda_mat_A, lambda_mat_C):
        self.lambda_mat_A = lambda_mat_A
        self.lambda_mat_C = lambda_mat_C
    def update(self, X, Y):
        # update based on the number
        w = self.lambda_mat_A
        scale = self.lambda_mat_C[self.is_server]
        alpha = self.A_C_alpha
        loss = scale * (1 - alpha) * F.cross_entropy(self.predict(X), Y).mean() + alpha * F.cross_entropy(self.predict(X), Y, weight = w).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}

class AFL(FedAvg):
    """
    Agnostic Federated Learning. (http://proceedings.mlr.press/v97/mohri19a/mohri19a.pdf)
    """

    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        super(AFL, self).__init__(input_shape, num_classes, hparams, is_server)
    def cal_w(self, client_list, client_tr, l):
        l_mat = torch.ones((1, self.hparams['n_client'])) * 1.0
        for id, data in enumerate (client_tr):
            loss_temp = self.evaluate_loss(data, client_list[id])
            l_mat[0, id] = len(data) * loss_temp * l[id]

        return l_mat
    def cal_lambda(self, client_list, client_tr):
        # l: 1 * n_client vector
        # weighted using the number of sample
        # sample -> weighted
        w_mat = torch.ones((1, self.hparams['n_client'])) * 1.0
        for id, data in enumerate(client_tr):
            loss_temp = self.evaluate_loss(data, client_list[id])
            w_mat[0, id] = len(data) * loss_temp
        return w_mat
    def server_update(self, client_list, client_tr):
        # updata the server
        # input: the list of clients
        # modify the self.network using our method
        # we use the training set of each client as the sampled data

        # step 4: update the weight vector
        
        gamma_w = self.hparams['gamma_w']
        gamma_l = self.hparams['gamma_l']
        T = self.hparams['T']

        # step 4.1: 
        w_mat = torch.ones((1, self.hparams['n_client'])) * 1.0 / self.hparams['n_client'] # T + 1 * n_client
        l_mat = torch.ones((1, self.hparams['n_client'])) * 1.0 / self.hparams['n_client']
        for t in range(T):
            delta_w = self.cal_w(client_list, client_tr, l_mat[t])
            delta_l = self.cal_lambda(client_list, client_tr)
            w_new = w_mat[t] - gamma_w * delta_w
            l_new = l_mat[t] + gamma_l * delta_l
            w_mat = torch.concat([w_mat, w_new], dim = 0)
            l_mat = torch.concat([l_mat, l_new], dim = 0)
        self.w = w_mat[1:].mean(dim = 0)
        self.l = l_mat[1:].mean(dim = 0)

        # step 1: get the parameters
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        print(self.w)
        # step 2: updata the model using the weight vector

        for index, key in enumerate(weight_keys):
            key_sum = 0
            for i in range(len(client_list)):
                key_sum = key_sum + worker_state_dict[i][key] * self.w[i]
            fed_state_dict[key] = key_sum
        # step 3: update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
        
class Q_FedAvg(FedAvg):
    """
    Q_FedAvg. (https://arxiv.org/pdf/1905.10497.pdf)
    """
    def __init__(self, input_shape, num_classes, hparams, is_server = 0):
        super(Q_FedAvg, self).__init__(input_shape, num_classes, hparams, is_server)
        self.w = torch.ones(hparams['n_client']) * 1.0 / hparams['n_client'] # will be updated
    def evaluate_loss_vec(self, client_list, te_dataset):
        # evaluate the performance of the client models on the corresponding dataset
        loss_mat = torch.zeros(len(te_dataset))
        for id_te, te in enumerate (te_dataset):#id_te == id_client
            loss = self.evaluate_loss(te, client_list[id_te])
            loss_mat[id_te] = loss
        return loss_mat
    def server_update(self, client_list, client_tr):
        # updata the server
        # input: the list of clients
        # modify the self.network using our method
        # we use the training set of each client as the sampled data
        # we implement the sampling process using re-weighting
        L = self.hparams["L_constant"]
        q = self.hparams["q"]
        # step 1: get the current weight of each client
        self.is_modified = 1
        worker_state_dict = [x.network.state_dict() for x in client_list]
        delta_w_state_dict = deepcopy(worker_state_dict) # delta_w = L(w_t - w^{_}_{t+1})
        delta_state_dict = deepcopy(worker_state_dict) # delta = delta_w * loss^q
        delta_h = torch.zeros(self.hparams['n_client']) # delta_h = q * loss^(q-1) * delta_w ^2
        L2_delta_w = torch.zeros(self.hparams['n_client'])
        server_state_dict = self.network.state_dict()
        weight_keys = list(worker_state_dict[0].keys())
        # step 2: get the loss of each client
        loss_vec = self.evaluate_loss_vec(client_list, client_tr)
        print("server update!")
        # step 3: subtract to get the delta_w
        for index, key in enumerate(weight_keys):
            for i in range(len(client_list)):
                delta_w_state_dict[i][key] = L * (server_state_dict[key] - worker_state_dict[i][key])
                delta_state_dict[i][key] = (loss_vec[i]**q) * delta_w_state_dict[i][key]
        for i in range(len(client_list)):
            L2_delta_w[i] = 0
            for index, key in enumerate(weight_keys):
                L2_delta_w[i] = L2_delta_w[i] + (delta_w_state_dict[i][key] * delta_w_state_dict[i][key]).sum()
        
        for i in range(len(client_list)):
            delta_h[i] = q * (loss_vec[i]**(q - 1)) * L2_delta_w[i] + L * (loss_vec[i]**q)
        # step 3: update server weights
        fed_state_dict = deepcopy(self.network.state_dict())
        
        # step 4: updata the model using the delta vectors
        for index, key in enumerate(weight_keys):
            
            key_sum = fed_state_dict[key]
            for i in range(len(client_list)):
                key_sum = key_sum - delta_state_dict[i][key] / delta_h.sum()
            fed_state_dict[key] = key_sum
        # step 3: update server weights
        self.network.load_state_dict(fed_state_dict)
        for model in client_list:
            model.network.load_state_dict(fed_state_dict)
    









        
    


        

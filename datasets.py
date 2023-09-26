import os
import torch
import random
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from torch.utils.data import Dataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
from FMDA.hparams import set_hparams
from sklearn.model_selection import train_test_split
from copy import deepcopy
DATASETS = [
    'Covid'
    ,'Support'
    ,'Cardio'
]
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
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    Divide n_client supgroups.
    '''

    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
 
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


class TabularData:
    '''
    Basic Class of the tabular data. For FL training.
    '''
    def __init__(self):
        self.tr_datasets = []
        self.te_datasets = []
        self.num_classes = 0
        self.input_shape = 0
    def get_train(self, index):
        return self.tr_datasets[index]
    def get_test(self, index):
        return self.te_datasets[index]
    def __len__(self):
        # return the number of clients
        return len(self.tr_datasets)
    def divide_data(self, X, Y, hparams, Sens = None, AX = None, AY = None):
        # X,Y:DataFrame
        self.divide_data_normal(X, Y, hparams, Sens)
        ############## for agnostic  ##########
        if hparams['agnostic'] == 1:
            self.divide_data_modify(AX, AY, hparams)
    def divide_data_modify(self, X, Y, hparams):
        self.num_classes = len(Y.unique())
        self.input_shape = len(X.columns)
        self.columns = X.columns
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y
                                                            , train_size = 1- hparams['holdout_fraction']
                                                            ,  test_size = hparams['holdout_fraction'])                     
        X_train = torch.tensor(np.array(X_train))
        X_test = torch.tensor(np.array(X_test))
        Y_train = torch.tensor(np.array(Y_train))
        Y_test = torch.tensor(np.array(Y_test))
        X_train = X_train.to(torch.float32)
        X_test = X_test.to(torch.float32)
        Y_train = Y_train.to(torch.long)
        Y_test = Y_test.to(torch.long)
        
        client_idcs = dirichlet_split_noniid(Y_train, hparams['alpha'], hparams['n_client'])

        self.total_te = (X_test, Y_test)
        for i in range(hparams['n_client']):
            
            temp_X = X_train[client_idcs[i]]
            temp_Y = Y_train[client_idcs[i]]
            XC_train, XC_test, YC_train, YC_test = train_test_split(temp_X, temp_Y
                                                            , train_size = 1- hparams['holdout_fraction']
                                                            ,  test_size = hparams['holdout_fraction'])
            self.te_datasets.append((XC_test, YC_test))
            
    def divide_data_modify_dif_alpha(self, X, Y, hparams):
        print("modify alpha")
        # modif0y the testing datasets
        self.num_classes = len(Y.unique())
        self.input_shape = len(X.columns)
        self.columns = X.columns
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y
                                                            , train_size = 1- hparams['holdout_fraction']
                                                            ,  test_size = hparams['holdout_fraction'])                     
        X_train = torch.tensor(np.array(X_train))
        X_test = torch.tensor(np.array(X_test))
        Y_train = torch.tensor(np.array(Y_train))
        Y_test = torch.tensor(np.array(Y_test))
        X_train = X_train.to(torch.float32)
        X_test = X_test.to(torch.float32)
        Y_train = Y_train.to(torch.long)
        Y_test = Y_test.to(torch.long)
        
        client_idcs = dirichlet_split_noniid(Y_train, hparams['agnostic_alpha'], hparams['n_client'])
        
        self.total_te = (X_test, Y_test)
        self.te_datasets = []
        for i in range(hparams['n_client']):
            temp_X = X_train[client_idcs[i]]
            temp_Y = Y_train[client_idcs[i]]
            XC_train, XC_test, YC_train, YC_test = train_test_split(temp_X, temp_Y
                                                            , train_size = 1- hparams['holdout_fraction']
                                                            ,  test_size = hparams['holdout_fraction'])
            
            self.te_datasets.append((XC_test, YC_test))
            print(len(XC_train), len(XC_test))
            print("test_client",i)
            test_data = myDataset(XC_test, YC_test,2)
            attr_list = test_data.split_by_attr()
            for idx,attr in enumerate(attr_list):
                print("attr", idx)
                print("lenth",len(attr))
            
        print("modified:",len(self.te_datasets))
    def divide_data_normal(self, X, Y, hparams, Sens = None):
        self.num_classes = len(Y.unique())
        self.input_shape = len(X.columns)
        self.columns = X.columns
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y
                                                            , train_size = 1- hparams['holdout_fraction']
                                                            ,  test_size = hparams['holdout_fraction'])
        
        # transform to tensor
        if Sens is None:
            Sens_train = deepcopy(Y_train)
        else:
            Sens_train = torch.tensor(np.array(X_train[Sens])).to(torch.long)
                                
        X_train = torch.tensor(np.array(X_train))
        X_test = torch.tensor(np.array(X_test))
        Y_train = torch.tensor(np.array(Y_train))
        Y_test = torch.tensor(np.array(Y_test))
        X_train = X_train.to(torch.float32)
        X_test = X_test.to(torch.float32)
        Y_train = Y_train.to(torch.long)
        Y_test = Y_test.to(torch.long)
        
        client_idcs = dirichlet_split_noniid(Sens_train, hparams['alpha'], hparams['n_client'])

        self.total_te = (X_test, Y_test)
        for i in range(hparams['n_client']):
            
            temp_X = X_train[client_idcs[i]]
            temp_Y = Y_train[client_idcs[i]]
            XC_train, XC_test, YC_train, YC_test = train_test_split(temp_X, temp_Y
                                                            , train_size = 1- hparams['holdout_fraction']
                                                            ,  test_size = hparams['holdout_fraction'])
            self.tr_datasets.append((XC_train, YC_train))
            self.te_datasets.append((XC_test, YC_test))
            print(len(XC_train))
            print(len(XC_test))
            print("test_client",i)
            test_data = myDataset(XC_test, YC_test,2)
            attr_list = test_data.split_by_attr()
            for idx,attr in enumerate(attr_list):
                print("attr", idx)
                print("lenth",len(attr))
            print("train_client",i)
            train_data = myDataset(XC_train, YC_train,2)
            attr_list = train_data.split_by_attr()
            for idx,attr in enumerate(attr_list):
                print("attr", idx)
                print("lenth",len(attr))

            
class Covid(TabularData):
    def __init__(self, data_dir, hparams):
        '''
        Load data from the local.
        Then we get the total test set.
        And then divide the data n_client parts.
        Then we get each part for each client.
        '''
        super().__init__()
        ################ loading and process data ################
        self.tr_datasets = []
        self.te_datasets = []
        all_data = self.read_data(data_dir) # 44 col (43 features, 1 label)
        X = all_data
        Y = all_data['is_dead']
        print(X.shape)
        X.to_csv("Covid_pro.csv")
        del X['is_dead']
        if hparams['test_EO'] == 1:
            print(X.head())
            print(X.columns)
            self.sens_attr = X.columns.tolist().index(hparams['sens_attr'])
            hparams['sens_index'] = self.sens_attr
        self.sens_attr = X.columns.tolist().index(hparams['sens_attr'])
        hparams['sens_index'] = self.sens_attr
        print("sens_index",hparams['sens_index'] )

        
        ################ divide dataset ################
        self.divide_data(X, Y, hparams)
        if hparams['agnostic_alpha'] > 0:
            self.divide_data_modify_dif_alpha(X, Y, hparams)
        
    def read_data(self, data_dir):
        # From the Covid-19-Brazil dataset
        path = data_dir + "INFLU20-04052020.csv"
        all_data = pd.read_csv(path, sep=';', encoding = "ISO-8859-1")
        all_data = all_data[all_data['PCR_SARS2']==1]
        all_data = all_data[~all_data['CS_RACA'].isnull()]
        all_data = all_data[all_data['CS_RACA']!=9]
        hospitalized_patients = all_data[(all_data['HOSPITAL']==1)]
        translate = {'NU_IDADE_N': 'Age', 'CS_SEXO': 'Sex', 'EVOLUCAO':'Evolution', 'CS_RACA':'Race',
    
             'FEBRE':'Fever', 'TOSSE': 'Cough', 'GARGANTA': 'Sore_throat', 
             'DISPNEIA':'Shortness_of_breath', 'DESC_RESP':'Respiratory_discomfort', 'SATURACAO':'SPO2',  
             'DIARREIA':'Dihareea', 'VOMITO':'Vomitting', 
             
             'CARDIOPATI': 'Cardiovascular', 'HEPATICA': 'Liver', 'ASMA': 'Asthma', 
             'DIABETES': 'Diabetis', 'NEUROLOGIC': 'Neurologic', 'PNEUMOPATI': 'Pulmonary',
             'IMUNODEPRE': 'Immunosuppresion', 'RENAL':'Renal', 'OBESIDADE': 'Obesity'}
        hospitalized_patients = hospitalized_patients.rename(columns=translate)
        demographics = ['Age', 'Sex', 'Race', 'SG_UF_NOT']

        symptoms = ['Fever','Cough', 'Sore_throat', 'Shortness_of_breath', 'Respiratory_discomfort', 'SPO2', 'Dihareea', 'Vomitting']

        comorbidities = ['Cardiovascular',  'Asthma', 'Diabetis', 'Pulmonary', 'Immunosuppresion',
                 'Obesity', 'Liver', 'Neurologic', 'Renal']

        outcome = ['Evolution']

        races = ['Branca', 'Preta', 'Amarela', 'Parda', 'Indigena']

        event_dates = ['DT_SIN_PRI', 'DT_COLETA', 'DT_PCR', 'DT_INTERNA',  'DT_ENTUTI', 'DT_EVOLUCA', 'DT_ENCERRA']

        age_groups = ['Age_40', 'Age_40_50', 'Age_50_60', 'Age_60_70', 'Age_70']
        race_encoding = {1.0: 'Branca', 2.0:'Preta', 3.0:'Amarela', 4.0:'Parda', 5.0: 'Indigena'}
        
        hospitalized_patients['Race'] = hospitalized_patients['Race'].apply(lambda i: race_encoding[i])
        ################ Add my modification on race##############
        
        transform_list = ['Race'# transfor to 0/1
                          ]
        for item in transform_list:
            uni = hospitalized_patients[item].unique()
            for col in uni:
                # initialize
                hospitalized_patients[uni] = 0
                
            for index in hospitalized_patients.index:
                for col in uni:
                    if hospitalized_patients.loc[index, item] == col:
                        hospitalized_patients.loc[index, col] = 1
                    else:
                        hospitalized_patients.loc[index, col] = 0
        ########################################################
        for race in races:
            hospitalized_patients[race] = (hospitalized_patients['Race'] == race).astype(np.int32)
        hospitalized_patients['Age_40'] = (hospitalized_patients['Age'] < 40).astype(np.int32)
        hospitalized_patients['Age_40_50'] = ((hospitalized_patients['Age'] >= 40) &  (hospitalized_patients['Age'] < 50)).astype(np.int32)
        hospitalized_patients['Age_50_60'] = ((hospitalized_patients['Age'] >= 50) &  (hospitalized_patients['Age'] < 60)).astype(np.int32)
        hospitalized_patients['Age_60_70'] = ((hospitalized_patients['Age'] >= 60) &  (hospitalized_patients['Age'] < 70)).astype(np.int32)
        hospitalized_patients['Age_70'] = (hospitalized_patients['Age'] >= 70).astype(np.int32)

        for event_date in event_dates:
            hospitalized_patients[event_date] = pd.to_datetime(hospitalized_patients[event_date], format="%d/%m/%Y")
        hospitalized_patients['Sex_male'] = (hospitalized_patients['Sex'] == 'M').astype(np.int32)
        hospitalized_patients['is_dead'] = (hospitalized_patients['Evolution'] == 2).astype(np.int32)

        for comorbidity in comorbidities:
            # fill in missing values for comorbidities
            hospitalized_patients[comorbidity][hospitalized_patients[comorbidity].isnull()] = 0
            hospitalized_patients[comorbidity][hospitalized_patients[comorbidity] == 9] = 0
            # 2 indicates in the data that the comorbidity is not present
            hospitalized_patients[comorbidity][hospitalized_patients[comorbidity] == 2] = 0

        for symptom in symptoms:
            # fill in missing values for symptoms
            hospitalized_patients[symptom][hospitalized_patients[symptom].isnull()] = 0
            hospitalized_patients[symptom][hospitalized_patients[symptom] == 9] = 0
            # 2 indicates in the data that the symptom is not present
            hospitalized_patients[symptom][hospitalized_patients[symptom] == 2] = 0
        hospitalized_patients['Days_hospital_to_outcome'] = (hospitalized_patients['DT_EVOLUCA'] - hospitalized_patients['DT_INTERNA']).dt.days
        patients_with_outcome = hospitalized_patients[~hospitalized_patients['Days_hospital_to_outcome'].isnull()]
        all_data = patients_with_outcome
        ################ Extra: delete the non-value type ################

        col_name = deepcopy(all_data.columns)
        col_type = deepcopy(all_data.dtypes)
        
        for i in range(col_type.shape[0]):
           
            if col_type[i] == "object" or col_type[i] == "datetime64[ns]" or col_type[i] == "str" or all_data[col_name[i]].isnull().any() == True:
                del all_data[col_name[i]]
        
        return all_data
class Support(TabularData):
    def __init__(self, data_dir, hparams):
        '''
        Load data from the local.
        Then we get the total test set.
        And then divide the data n_client parts.
        Then we get each part for each client.
        '''
        super().__init__()
        ################ loading and process data ################
        self.tr_datasets = []
        self.te_datasets = []
        all_data = self.read_data(data_dir) # 100 col (99 features, 1 label)
        
        X = all_data
        Y = all_data['death']
        X.to_csv("Support_pro.csv")
        del X['death']
        print(X.shape)

        if hparams['test_EO'] == 1:
            print(X.head())
            print(X.columns)
            self.sens_attr = X.columns.tolist().index(hparams['sens_attr'])
            hparams['sens_index'] = self.sens_attr
        ################ divide dataset ################
        self.divide_data(X, Y, hparams)   
    
    def read_data(self, data_dir):
        # From the Covid-19-Brazil dataset
        path = data_dir + "support.xlsx"
        all_data = pd.read_excel(path)
        ################ clean the dataset ################
        # transform to 0-1 and remove the almost-NaN features
        feature_list = [  'death', 'age', 'sex', 'num.co'
                        , 'edu', 'avtisst', 'meanbp', 'wblc'
                        , 'hrt', 'resp', 'temp', 'pafi'
                        , 'alb', 'bili', 'crea', 'sod'
                        , 'ph', 'glucose', 'bun', 'urine'
                        , 'adlp', 'adls' # value
                        ]
        transform_list = ['income', 'dzgroup', 'dzclass', 'race'# transfor to 0/1
                          ]
        
        # transform to one-hot code
        for item in transform_list:
            uni = all_data[item].unique()
            for col in uni:
                # initialize
                all_data[col] = 0
                feature_list.append(col)
            for index in range(all_data.shape[0]):
                for col in uni:
                    if all_data.loc[index, item] == col:
                        all_data.loc[index, col] = 1
                    else:
                        all_data.loc[index, col] = 0
        
        for index in range(all_data.shape[0]):
            if all_data.loc[index, 'sex'] == "female":
                all_data.loc[index, 'sex'] = 0
            else:
                all_data.loc[index, 'sex'] = 1
        
        all_data = all_data[feature_list]
        
        col_list = all_data.columns
        for col in col_list:
            flag = 0
            for item in all_data[col].isnull():
                if item == True:
                    flag = 1
                    break
            if flag == 1:
                del all_data[col]

        if "nan" in all_data.columns:
            del all_data["nan"]
        if "sex" in all_data.columns:
            del all_data["sex"]
        all_data = all_data.dropna(how = "any")
        
        
        return all_data
class Support_Ind(Support):
    def __init__(self, data_dir, hparams):
        '''
        Load data from the local.
        Then we get the total test set.
        And then divide the data n_client parts.
        Then we get each part for each client.
        '''
        super(Support_Ind, self).__init__(data_dir, hparams)
        ################ loading and process data ################
        self.tr_datasets = []
        self.te_datasets = []
        all_data = self.read_data(data_dir) # 100 col (99 features, 1 label)
        
        X = all_data
        Y = all_data['death']
        del X['death']
        print(X.shape)
        X['id'] = 0 # act as a identification
        for i in range(X.shape[0]):
            X.loc[i, 'id'] = i
        print(X.columns)
        if hparams['test_EO'] == 1:
            print(X.head())
            print(X.columns)
            self.sens_attr = X.columns.tolist().index(hparams['sens_attr'])
            hparams['sens_index'] = self.sens_attr
        ################ divide dataset ################
        self.divide_data(X, Y, hparams)   
    
class Cardio(TabularData):
    def __init__(self, data_dir, hparams):
        '''
        Load data from the local.
        Then we get the total test set.
        And then divide the data n_client parts.
        Then we get each part for each client.
        '''
        super().__init__()
        ################ loading and process data ################
        self.tr_datasets = []
        self.te_datasets = []
        all_data = self.read_data(data_dir) # 100 col (99 features, 1 label)
        X = all_data
        Y = all_data['Class']
        for i in range(Y.shape[0]):
            Y[i] = Y[i] - 1
        X.to_csv("Cardio_pro.csv")
        del X['Class']
        print(X.shape)
        ################ divide dataset ################
        self.divide_data(X, Y, hparams)  
    def read_data(self, data_dir):
        # From the Covid-19-Brazil dataset
        path = data_dir + "cardiotocography_csv.csv"
        all_data = pd.read_csv(path)
        all_data = all_data.dropna(how = "any")
        
        return all_data
class Cardio_Ind(Cardio):
    def __init__(self, data_dir, hparams):
        '''
        Load data from the local.
        Then we get the total test set.
        And then divide the data n_client parts.
        Then we get each part for each client.
        '''
        super(Cardio_Ind, self).__init__(data_dir, hparams)
        ################ loading and process data ################
        self.tr_datasets = []
        self.te_datasets = []
        all_data = self.read_data(data_dir) # 100 col (99 features, 1 label)
        X = all_data
        Y = all_data['Class']
        for i in range(Y.shape[0]):
            Y[i] = Y[i] - 1
        
        del X['Class']
        print(X.shape)
        X['id'] = 0 # act as a identification
        for i in range(X.shape[0]):
            X.loc[i, 'id'] = i
        ################ divide dataset ################
        self.divide_data(X, Y, hparams)  
    
class SEER(TabularData):
    def __init__(self, data_dir, hparams):
        '''
        Load data from the local.
        Then we get the total test set.
        And then divide the data n_client parts.
        Then we get each part for each client.
        '''
        super().__init__()
        ################ loading and process data ################
        # step 1: get the Label list!!! important: avoid missing + alignment
        all_data, A_data = self.read_data(data_dir, hparams) # 100 col (99 features, 1 label)
        X = all_data
        Y = all_data['Site recode ICD-O-3/WHO 2008']
        uni_Y = Y.unique().tolist() # for reference
        Y = Y.reset_index()
        Y = Y.iloc[:, 1]
        
        # set the smaller one as "others"
        refer_Y = deepcopy(uni_Y)
        for index, value in enumerate(Y.value_counts()):
            item = Y.value_counts().index[index]
            
            if index >= 10:
                index = 10
            refer_Y[uni_Y.index(item)] = index
        
        ###########################################################
        # step 2: load daa
        self.tr_datasets = []
        self.te_datasets = []
        all_data, A_data = self.read_data(data_dir, hparams) # 100 col (99 features, 1 label)
        
        def split_X_Y(all_data, uni_Y, refer_Y):
            X = all_data
            Y = all_data['Site recode ICD-O-3/WHO 2008'] 
            Y_list = Y.reset_index()
            Y_list = Y_list.iloc[:, 1]
            Y = pd.Series(np.zeros(Y.shape[0]))
            for i in range(Y.shape[0]):
                Y[i] = refer_Y[uni_Y.index(Y_list[i])]
            Y = Y.astype("int")
            X.to_csv("SEER.csv")
            del X['Site recode ICD-O-3/WHO 2008']
            del X['Unnamed: 0']
            return X, Y
        X, Y = split_X_Y(all_data, uni_Y, refer_Y)
        print(X.shape)
        self.sens_attr = X.columns.tolist().index(hparams['sens_attr'])
        hparams['sens_index'] = self.sens_attr
        ################ divide dataset ################
 
        if hparams['agnostic'] == 1:
            AX, AY = split_X_Y(A_data, uni_Y, refer_Y)
            self.divide_data(X, Y, hparams, None, AX, AY)
        else:
            self.divide_data(X, Y, hparams) 
            if hparams['agnostic_alpha'] > 0:
                self.divide_data_modify_dif_alpha(X, Y, hparams)
    def read_data(self, data_dir, hparams):
        path = data_dir + "seer"
        
        # get the columns
        is_load = 0
        for foldername, subfolders, filenames in os.walk(path):
            for filename in filenames:
                
                if "xlsx" in filename:
                    if filename.replace(".xlsx", ".csv") in filenames:
                        read_path = os.path.join(path, filename.replace(".xlsx", ".csv"))
                        
                        temp_data = pd.read_csv(read_path)
                        
                    else:
                        temp_data = self.read_data_process(data_dir, filename)
                else:
                    continue
                    
                if is_load == 0:
                    is_load = 1
                    all_columns = temp_data.columns.tolist()
                else:
                    all_columns = all_columns + temp_data.columns.tolist()
        
        all_columns = list(pd.Series(all_columns).unique())
        print("Length of the columns:",len(all_columns))
        # get the dataset
        is_load = 0
        for foldername, subfolders, filenames in os.walk(path):
            file_len = 0
            for index, filename in enumerate(filenames):
                if "csv" in filename:
                    file_len = file_len + 1
            
            count = 0
            for index, filename in enumerate(filenames):
                
                
                if "csv" in filename:
                    read_path = os.path.join(path, filename)
                        
                    temp_data = pd.read_csv(read_path)
                    count = count + 1
                        
                else:
                    continue
                

                if is_load == 0:
                    is_load = 1
                    # check if the columns are in all_columns
                    for col in temp_data.columns:
                        if col not in all_columns:
                            print("help")
                    
                    # set the missing columns
                    for col in all_columns:
                        if col not in temp_data.columns:
                            
                            temp_data[col] = 0
                    all_data = temp_data
                    
                else:
                    for col in temp_data.columns:
                        if col not in all_columns:
                            print("help")
                    
                    # set the missing columns
                    for col in all_columns:
                        if col not in temp_data.columns:
                            
                            temp_data[col] = 0
                    if not(count == file_len - 1) or not(hparams['agnostic'] == 1):
                        all_data = pd.concat([all_data, temp_data], axis = 0)
                    else:
                        print("agnostic:", filename)
        if hparams['agnostic'] == 1:
            
            A_data = temp_data
        else:
            A_data = None
        all_data = all_data.dropna(how = "any")
        
        return all_data, A_data
    def read_data_process(self, data_dir, year_name):
        # iter all files and load data
        # read a single year's data
        path = data_dir + "seer"
        is_load = 0

        file_name = os.path.join(path, year_name)
        for foldername, subfolders, filenames in os.walk(path):
            for filename in filenames:
                if not (filename == year_name):
                    continue
                
                temp_data = pd.read_excel(os.path.join(path, filename))
                
                temp_data = temp_data.dropna(how = "any")
                all_data = temp_data
                break    
        print("read done!")
        print(all_data.columns)
        
        feature_list = [
            'Age recode with <1 year olds'
            ,'Race recode (White, Black, Other)'
            ,'Sex'
            ,'PRCDA 2020'
            ,'Site recode ICD-O-3/WHO 2008'
            
            ,'Primary Site'
            ,'Histologic Type ICD-O-3'
            ,'Behavior recode for analysis'
            
            ,'Laterality'
            ,'Diagnostic Confirmation'
            
            ,'RX Summ--Surg Prim Site (1998+)'
            ,'Reason no cancer-directed surgery'
            ,'RX Summ--Systemic/Sur Seq (2007+)'
        
        ]
        transform_list = deepcopy(feature_list)
        transform_list.remove('Primary Site')
        transform_list.remove('Histologic Type ICD-O-3')
        transform_list.remove('Site recode ICD-O-3/WHO 2008')
        transform_list.remove('RX Summ--Surg Prim Site (1998+)')
        
        for item in transform_list:
            feature_list.remove(item)
        for item in transform_list:
            uni = all_data[item].unique()
            for col in uni:
                # initialize
                all_data[col] = 0
                feature_list.append(col)
                def select(x, col, item):
                    if x[item] == col:
                        return 1
                    else:
                        return 0
                all_data[col] = all_data.apply(lambda x: select(x, col, item), axis = 1)

        
        
        all_data = all_data[feature_list]
        all_data.to_csv(os.path.join(path, filename.replace(".xlsx", ".csv")))
        
        return all_data
class myDataset(Dataset):
    '''
    For data loading.
    '''
    def __init__(self, X, Y, num_splits):
        super().__init__()
        self.X = X
        self.Y = Y
        self.num_splits = num_splits
        
    def __getitem__(self, index):
        return (self.X[index], self.Y[index])
    def split_by_attr(self, index = -1):
        # return a list of datasets
        # index: feature.-1: label
        
        
        X_list = []
        Y_list = []
        for i in range(self.num_splits):
            X_list.append([])
            Y_list.append([])
        
        for i in range(self.X.shape[0]):
            
            X_list[self.Y[i]].append(self.X[i])
            Y_list[self.Y[i]].append(self.Y[i])
        dataset_list = []
        for i in range(self.num_splits):
            dataset_list.append(myDataset(X_list[i],Y_list[i],self.num_splits))
        return dataset_list
    def split_by_attr_22(self, num_attr, sens_index, index = -1):
        # return a list of datasets
        # index: feature.-1: label
        
        
        X_list = []
        Y_list = []
        for i in range(num_attr):
            X_list.append([])
            Y_list.append([])
        
        for i in range(self.X.shape[0]):
            if self.X[i, sens_index] == 1:
                index_temp = self.Y[i] + 11
            else:
                index_temp = self.Y[i]
            X_list[index_temp].append(self.X[i])
            Y_list[index_temp].append(self.Y[i])
            
        dataset_list = []
        for i in range(num_attr):
            
            dataset_list.append(myDataset(X_list[i],Y_list[i],self.num_splits))
        return dataset_list
    def split_by_attr_2(self, num_attr, sens_index, index = -1):
        # return a list of datasets
        # index: feature.-1: label
        
        
        X_list = []
        Y_list = []
        for i in range(num_attr):
            X_list.append([])
            Y_list.append([])
        
        for i in range(self.X.shape[0]):
            if self.X[i, sens_index] == 1:
                index_temp = 1
            else:
                index_temp = 0
            X_list[index_temp].append(self.X[i])
            Y_list[index_temp].append(self.Y[i])
            
        dataset_list = []
        for i in range(num_attr):
            dataset_list.append(myDataset(X_list[i],Y_list[i],self.num_splits))
        return dataset_list
    def split_by_attr_overall(self, index = -1):
        # return a list of datasets
        # index: feature.-1: label
        
        
        X_list = []
        Y_list = []
        for i in range(self.num_splits):
            X_list.append([])
            Y_list.append([])
        
        for i in range(self.X.shape[0]):
            
            X_list[self.Y[i]].append(self.X[i])
            Y_list[self.Y[i]].append(self.Y[i])
        number_vector = np.zeros(len(X_list))
        for i in range(len(X_list)):
            number_vector[i] = len(X_list[i])
        dataset_list = []
        for i in range(self.num_splits):
            dataset_list.append(myDataset(X_list[i],Y_list[i],self.num_splits))
        return dataset_list, number_vector
    def split_by_attr_22(self, num_attr, sens_index, index = -1):
        # return a list of datasets
        # index: feature.-1: label
        
        
        X_list = []
        Y_list = []
        for i in range(num_attr):
            X_list.append([])
            Y_list.append([])
        
        for i in range(self.X.shape[0]):
            if self.X[i, sens_index] == 1:
                index_temp = self.Y[i] + 11
            else:
                index_temp = self.Y[i]
            X_list[index_temp].append(self.X[i])
            Y_list[index_temp].append(self.Y[i])
        number_vector = np.zeros(len(X_list))
        for i in range(len(X_list)):
            number_vector[i] = len(X_list[i])
        dataset_list = []
        dataset_list = []
        for i in range(num_attr):
            
            dataset_list.append(myDataset(X_list[i],Y_list[i],self.num_splits))
        return dataset_list, number_vector
    def split_by_attr_4(self, num_attr, sens_index, index = -1):
        # return a list of datasets
        # index: feature.-1: label
        
        
        X_list = []
        Y_list = []
        for i in range(num_attr):
            X_list.append([])
            Y_list.append([])
        
        for i in range(self.X.shape[0]):
            if self.X[i, sens_index] == 1:
                index_temp = self.Y[i] + 2
            else:
                index_temp = self.Y[i]
            X_list[index_temp].append(self.X[i])
            Y_list[index_temp].append(self.Y[i])
        number_vector = np.zeros(len(X_list))
        for i in range(len(X_list)):
            number_vector[i] = len(X_list[i])
        dataset_list = []
        dataset_list = []
        for i in range(num_attr):
            
            dataset_list.append(myDataset(X_list[i],Y_list[i],self.num_splits))
        return dataset_list, number_vector
    def split_by_attr_2(self, num_attr, sens_index, index = -1):
        # return a list of datasets
        # index: feature.-1: label
        
        
        X_list = []
        Y_list = []
        for i in range(num_attr):
            X_list.append([])
            Y_list.append([])
        
        for i in range(self.X.shape[0]):
            if self.X[i, sens_index] == 1:
                index_temp = 1
            else:
                index_temp = 0
            X_list[index_temp].append(self.X[i])
            Y_list[index_temp].append(self.Y[i])
        number_vector = np.zeros(len(X_list))
        for i in range(len(X_list)):
            number_vector[i] = len(X_list[i])
        dataset_list = []
        dataset_list = []
        for i in range(num_attr):
            dataset_list.append(myDataset(X_list[i],Y_list[i],self.num_splits))
        return dataset_list, number_vector
    def __len__(self):
        return len(self.X)
    
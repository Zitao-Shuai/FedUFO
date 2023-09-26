Official Implementation of Cell Press Patterns Submission "Unified Fair Federated Learning for Digital Healthcare"

TL;DR: Proposing and achieving unified fairness on federated learning for digital healthcare based on distributionally robust optimization.

The running environment:

```shell
Python: 3.8.10
PyTorch: 1.13.1+cu117
Torchvision: 0.14.1+cu117
CUDA: 11.7
CUDNN: 8500
NumPy: 1.21.2
PIL: 9.3.0
```

For running the project:
Example 1: 

```shell
python3 -m FMDA.train\
	    --data_dir ./FMDA/data/\
        --algorithm FMDA_M_N\
        --dataset Covid\
        --type M\
        --alpha 0.5
```

Example 2:

```shell
python3 -m FMDA.train\
	    --data_dir ./FMDA/data/\
        --algorithm FedAvg\
        --dataset Covid\
        --test_EO 1\
        --alpha 1
```

Example 3:

```shell
python3 -m FMDA.train\
	    --data_dir ./FMDA/data/\
        --algorithm FMDA\
        --dataset SEER\
        --agnostic_alpha 1\
        --alpha 10
```

For datasets that are not provided due to the privacy issue, please apply for authorization through these links:

```
Support: https://biostat.app.vumc.org/wiki/Main/DataSets
Cardio(Fetal): https://datahub.io/machine-learning/cardiotocography
SEER(Prostate): https://seer.cancer.gov/data/
```

 

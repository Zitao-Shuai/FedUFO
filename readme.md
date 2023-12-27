This is an Official Implementation of FedUFO.

Environment:

```shell
Python: 3.8.10
PyTorch: 1.13.1+cu117
Torchvision: 0.14.1+cu117
CUDA: 11.7
CUDNN: 8500
NumPy: 1.21.2
PIL: 9.3.0
```

To run FedUFO on your own dataset：

Example 1: 

```shell
python3 -m FedUFO.train\
	    --data_dir ./FedUFO/data/\
        --algorithm FedUFO_M_N\
        --dataset Your_Own_Dataset\
        --type M\
        --alpha 0.5
```

Example 2:

```shell
python3 -m FedUFO.train\
	    --data_dir ./FedUFO/data/\
        --algorithm FedAvg\
        --dataset Your_Own_Dataset\
        --test_EO 1\
        --alpha 1
```

Example 3:

```shell
python3 -m FedUFO.train\
	    --data_dir ./FedUFO/data/\
        --algorithm FedUFO\
        --dataset Your_Own_Dataset\
        --agnostic_alpha 1\
        --alpha 10
```

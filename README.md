# Deep learning phase unwrapping with a U-Net
A demonstrate of "Deep learning spatial phase unwrapping: a comparative review" (in progress).

## Preparation
1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Datasets
Download dataset (without noise) from [here](https://figshare.com/s/685e972475221aa3b4c4) to the current path and unzip. The file structure is the following:
```
train_in
└── 000001.mat
...
└── 020000.mat
train_gt
└── 000001.mat
...
└── 020000.mat
test_in
└── 000001.mat
...
└── 002000.mat
test_gt
└── 000001.mat
...
└── 000421.mat
test_in_real
└── 000001.mat
...
└── 000421.mat
test_gt_real
└── 000001.mat
...
└── 000421.mat
```

Of course, datasets _train_in_, _train_gt_, _test_in_ and _test_gt_ can also be obtained by running _dataset_generation.m_ with MATLAB , whose parameters can be adjusted according to actual needs. (The size range of the noise can be controlled by adjusting the parameter _noise_max_.)

## Network traning
Run _main_train.py_ to start training the neural network.
```sh
python main_train.py
```
After training, two files (loss and others.csv and weights.pth) will be saved in the folder model_weights

## Network testing
Run _main_test.py_ to do some test.
```sh
python main_test.py
```

## Error statistics
Run _error_evaluation.m_ with MATLAB to calculate RMSEs for each test result.


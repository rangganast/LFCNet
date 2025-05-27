# Official Code of LFCNet

## Installation
we run the project on PyTorch==2.0.0 on CUDA 11.8 and Python==3.10.12

#### **Step 1.** Clone and install requirements
```
git clone https://github.com/rangganast/LFCNet.git
cd LFCNet
pip install -r requirements.txt
```

#### **Step 2.** Install Pytorch-Correlation-extension
```
cd ..
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension
python3 setup.py install
```

#### **Step 3.** Download the weights

- [Err: 1.5 | 20](https://drive.google.com/file/d/1f0Rze3AhPeZMfkDD7xzvytUM0MXsbAhE/view?usp=drive_link)
- [Err: 1.0 | 10](https://drive.google.com/file/d/1Z2TRimwVK6K2MFYu6Qce5jpIdgoEP4eI/view?usp=drive_link)
- [Err: 0.5 | 5](https://drive.google.com/file/d/1MuM5HA3OGUo-BL6-h9e8o9-14rgHrhzo/view?usp=drive_link)
- [Err: 0.2 | 2](https://drive.google.com/file/d/1HY0ALZVg5jrvfET3--jpXKgsjrcXtE4e/view?usp=drive_link)

#### Data Preparation
The data folders are organized as follows:
```
├── KITTI-360/
|   └── calibration
|   └── data_2d_raw
|        └── 2013_05_28_drive_0000_sync  
|            └── data
|               └── 000000.png
|               └── 000001.png
|               └── 000002.png
|               └── ...
|   └── data_3d_raw
|       └── 2013_05_28_drive_0000_sync  
|           └── data
|               └── 000000.bin
|               └── 000001.bin
|               └── 000002.bin
|               └── ...
```


## Testing
Put the weights in `checkpoint_weights/run`, and run:
```
python test_iterative.py
```
For iterative refinement evaluation, and
```
python test_continuous.py
```
For multi frame analysis

# Contact
For questions about our paper or code, please contact rangganast@gmail.com.

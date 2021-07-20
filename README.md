# 3D-BrainSeg with multiple modalities

This is the implementation of our method in the MICCAI Grand Challenge on [6-month infant brain MRI segmentation-in conjunction with MICCAI 2019](http://iseg2019.web.unc.edu) in Pytorch. 

### Introduction
6-month infant brain MRI segmentation aims to segment the brain into: White matter, Gray matter, and Cerebrospinal fluid. It is a difficult task due to larger overlapping between tissues, low contrast intensity. We treat the problem by using very deep 3D convolution neural network. Our result achieved the top performance in 6 performance metrics. 

### Citation
```
```

### Requirements: 
- Pytorch >=0.4, python 3.7, Ubuntu 18.04
- TiTan X Pascal 12GB

### Installation
- Step 1: Download the source code
```

cd 3D-SkipDenseSeg
```
- Step 2: Download dataset at `http://iseg2019.web.unc.edu/download/` and change the path of the dataset `data_path` and saved path `target_path` in file `prepare_hdf5_cutedge.py`
```
data_path = '/path/to/your/dataset/'
target_path = '/path/to/your/save/hdf5 folder/'
```

- Step 3: Generate hdf5 dataset

```
python prepare_hdf5_cutedge.py
```

- Step 4: Run training

```
python train.py
```

Run evaluation result. 
```
python val.py
```
We also provide pretrained model. Use the pretrained model, you should achieve the result as the table. 
### Dice Coefficient (DC) for 9th subject (9 subjects for training and 1 subject for validation)
|                   | Pretrained |  CSF       | GM             | WM   | Average 
|-------------------|:-------------------:|:-------------------:|:---------------------:|:-----:|:--------------:|
|3DBrainSeg  |  2 | 94.96 | 91.78 | 91.24 | 92.66 |


Run on testing set
```
python test.py
```


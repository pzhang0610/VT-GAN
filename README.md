# VT-GAN
An Pytorch implementation of VT-GAN for gait recognition

## Requirements
* pytorch v1.0.0
* PIL
* matplotlib
* numpy
* logging
* shutil

## Dataset

Download the CASIA gait dataset B from \url{http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp} and put it in directory './data'.
## Usage

For training, run

```
python run_triplet_block.py --is_train --batch_size 100 --resume_iters None
```

For generation, run

```
python run_triplet_block.py --batch_size 1
```
## Examples

# VT-GAN
An Pytorch implementation of VT-GAN for gait recognition

## Requirements
* pytorch v1.0.0
* PIL
* matplotlib
* numpy
* logging
* shutil

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

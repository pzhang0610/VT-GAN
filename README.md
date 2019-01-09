# VT-GAN
An Pytorch implementation of VT-GAN for gait recognition. The framework is as follow,

<img src="./model/framework.png" width = "600" height = "300" alt="Framework of VT-GAN" align=center />

## Requirements
* pytorch v1.0.0
* PIL
* matplotlib
* numpy
* logging
* shutil

## Dataset

Download the CASIA gait dataset B from http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp and put it in directory './data'.
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

Attached are some examples, the first column are inputs, the first row are references, and the bottom 11 x 11 are generated gaits,

<img src="./sample/sample.png" width = "500" height = "500" alt="An example of generated gaits" align=center />


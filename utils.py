import numpy as np
import os


def transform(img):
    return np.array(img)/127.5 - 1.


def inverse_transform(img):
    return (img + 1.)/2.


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

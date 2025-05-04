import numpy as np
from skimage.io import imread

mask = imread("./data/train/0aaa252e-b503-4503-bdc6-387a5cfe2622/class1.tif")
print("Unique values in mask:", np.unique(mask))

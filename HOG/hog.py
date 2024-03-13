import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature, exposure
img = plt.imread('japanese_vector_background.jpg', cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Original image size: ", img.shape)

# define parameters
cell_size = (8,8)
block_size = (2,2)
nbins = 9

# calculate parameters to HOGDescriptor
# winSize: the size of image is cropped to be divisible by the cell size
winSize = (img.shape[1]//cell_size[1]*cell_size[1], img.shape[0]//cell_size[0]*cell_size[0])
print("winSize: ", winSize)
# Block size
blockSize = (block_size[1]*cell_size[1],block_size[0]*cell_size[0])
blockStride = (cell_size[1], cell_size[0])

print("blockSize", blockSize)
print("blockStride", blockStride)

hog =  cv2.HOGDescriptor(
  _winSize= winSize,
  _blockSize= blockSize,
  _blockStride= blockStride,
  _cellSize= cell_size,
  _nbins= nbins
)

n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
print("n_cells ", n_cells)

hog_feats = hog.compute(img).reshape(
  n_cells[1] - block_size[1] +1,
  n_cells[0] - block_size[0] +1,
  block_size[0], block_size[1], nbins
).transpose((1, 0, 2, 3, 4))

print("hog_feats shape: ", hog_feats.shape)

(H, hogImage) = feature.hog(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                            orientations=9,
                            pixels_per_cell=cell_size,
                            cells_per_block=block_size,
                            transform_sqrt=True,
                            block_norm="L2",
                            visualize=True)

hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

plt.imshow(hogImage)
plt.show()


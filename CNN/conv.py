import numpy as np
import matplotlib.pyplot as plt
import cv2 
from PIL import Image
import urllib.request
from io import BytesIO

# %matplotlib inline


# X = np.array(Image.open(f))
# print("Image shape: ", X.shape)
# X = X.dot([0.299, 0.587, .114])
# print("Image shape: ", X.shape)
# plt.imshow(X)
# plt.show()

# Define F1 filter


# Define function to calculate convolution 2D
def conv2d(X, F, s = 1, p = 0):
  """
  params:
    X: Input matrix
    F: Filter matrix
    s: Stride
    p: Padding
  output:
    Y: Output matrix
  """
  (w1, h1) = X.shape
  f = F.shape[0]
  w2 = int((w1+ 2*p - f)/s) + 1
  h2 = int((h1+ 2*p - f)/s) + 1
  Y = np.zeros((w2, h2))
  X_pad = np.pad(X, pad_width=p, mode='constant', constant_values=0)
  for i in range(w2):
    for j in range(h2):
      Y[i, j] = np.sum(X_pad[i*s:i*s+f, j*s:j*s+f] * F)
  return Y

if __name__ == '__main__':
  url = str("https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/bbe48fc8-2001-4984-bcbd-39def882d9d1/dgy3e9h-57abbe6b-506f-49bc-b4e6-6705235ae31c.png/v1/fill/w_894,h_894,q_70,strp/gjidf45_7vms_by_joear1zz_dgy3e9h-pre.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTAyNCIsInBhdGgiOiJcL2ZcL2JiZTQ4ZmM4LTIwMDEtNDk4NC1iY2JkLTM5ZGVmODgyZDlkMVwvZGd5M2U5aC01N2FiYmU2Yi01MDZmLTQ5YmMtYjRlNi02NzA1MjM1YWUzMWMucG5nIiwid2lkdGgiOiI8PTEwMjQifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.2H2AHIfEeJF6x-iHjIE3ubikV5eORa7LcqwlPCXyt4c")
  with urllib.request.urlopen(url) as url:
    f = BytesIO(url.read())
  X = np.array(Image.open(f))
  print("Image shape: ", X.shape)
  X = X.dot([0.299, 0.587, .114])
  print("Image shape: ", X.shape)
  F1 = np.array([[-1, -1, -1,], [0, 0, 0], [1, 1, 1]])

  Y1 = conv2d(X, F1)
  plt.imshow(Y1)
  plt.title("Y1")


  F2 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
  Y2 = conv2d(X, F2, s = 3, p = 0)
  plt.imshow(Y2)
  plt.title("Y2")
  plt.show()
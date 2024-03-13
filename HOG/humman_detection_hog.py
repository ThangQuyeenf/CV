from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

images = glob.glob("images\*.png")
print(len(images))
for i, img_path in enumerate(images):
  img = cv2.imread(img_path)
  img = imutils.resize(img, width= min(400, img.shape[1]))
  orig = img.copy()
  
  plt.figure(figsize=(10,6))

  ax1 = plt.subplot(1,2,1)

  # detection
  (rects, weights) =hog.detectMultiScale(img=img, winStride=(4,4), padding=(8,8), scale=1.05)
  print("weights", weights)

  # draw bounding box
  for (x,y,w,h) in rects:
    #cv2.rectangle(orig,(x,y),(x+w,y+h),(255,0,0),2)
    rectFig = patches.Rectangle((x,y),w,h, linewidth=1, edgecolor='r', facecolor='none')
    ax1.imshow(orig)
    ax1.add_patch(rectFig)
    plt.title('Before none max suppression')
    

  rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
  print('rects: ', rects.shape)

  # applies non max suppression with threshold 0.65
  pick = non_max_suppression(rects, probs = None, overlapThresh=0.65)

  ax2 = plt.subplot(1,2,2)

  for (x1, y1, x2, y2) in pick:
    w = x2 - x1
    h = y2 - y1

    plt.imshow(img)
    plt.title('After non max suppression')
    rectFig = patches.Rectangle((x1,y1),w,h, linewidth=1, edgecolor='r', facecolor='none')
    ax2.add_patch(rectFig)

  filename = img_path[img_path.rfind("\\") + 1:]
  print("[INFO] {}: {} original boxes, {} after suppression".format(
      filename, len(rects), len(pick)))
  
  plt.show()
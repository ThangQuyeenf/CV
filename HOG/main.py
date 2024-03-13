import numpy as np
import cv2
import matplotlib.pyplot as plt

img = plt.imread('japanese_vector_background.jpg', cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('image shape:', img.shape)
print('gray shape: ', gray.shape)

## Show image
# plt.figure(figsize = (16, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.title('Original Image')
# plt.subplot(1, 2, 2)
# plt.imshow(gray)
# plt.title('Gray Image')
# plt.show()

# Calculate gradient
gx = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=3)

g, theta = cv2.cartToPolar(gx, gy, angleInDegrees=True)

print("Gray shape {}".format(gray.shape))
print("gx shape {}".format(gx.shape))
print("gy shape {}".format(gy.shape))

plt.figure(figsize=(20,10))
plt.subplot(2, 2, 1)
plt.imshow(gx)
plt.title("gradient of x")

plt.subplot(2, 2, 2)
plt.imshow(gy)
plt.title("gradient of y")

plt.subplot(2, 2, 3)
plt.imshow(theta)
plt.title("Direction of gradient")

plt.subplot(2, 2, 4)
plt.imshow(theta)
plt.title("Magnitude of gradient")
plt.suptitle("Sobel gradient")
plt.show()


gx1 = cv2.Scharr(gray, cv2.CV_32F, dx=0, dy=1, scale=2)
gy1 = cv2.Scharr(gray, cv2.CV_32F, dx=1, dy=0, scale=2)

g1, theta1 = cv2.cartToPolar(gx1, gy1, angleInDegrees=True)

print("Gray shape {}".format(gray.shape))
print("gx shape {}".format(gx1.shape))
print("gy shape {}".format(gy1.shape))

plt.figure(figsize=(20,10))
plt.subplot(2, 2, 1)
plt.imshow(gx1)
plt.title("gradient of x")

plt.subplot(2, 2, 2)
plt.imshow(gy1)
plt.title("gradient of y")

plt.subplot(2, 2, 3)
plt.imshow(theta1)
plt.title("Direction of gradient")

plt.subplot(2, 2, 4)
plt.imshow(theta1)
plt.title("Magnitude of gradient")
plt.suptitle("Scharr gradient")
plt.show()
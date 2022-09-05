import cv2
from skimage.transform import resize

path = "/home/fmasa/Pictures/test_aisle.png"

img = cv2.imread(path)

img2 = resize(img, (256, 256), mode='constant')

cv2.imshow("window", img)
cv2.imshow("resize", img2)
cv2.waitKey(0)

import numpy
import cv2

print("NumPy OK")
print("OpenCV OK")
exit()
import cv2
import numpy as np

print("AI Face Photo Finder")
print("Python setup is working!")

image = cv2.imread("images/test.jpg")

if image is None:
    print("Image not found")
else:
    print("Image loaded successfully!")
    print("Image shape:", image.shape)
import cv2
import numpy as np

width, height = 900, 600
img = np.zeros((height, width, 3), np.uint8)

# Ellipse parameters
radius = 50
center = (100, 100)
axes = (radius, radius)
angle = 0
startAngle = 0
endAngle = -43
thickness = 10
WHITE = (255, 255, 255)

cv2.ellipse(img, center, axes, angle, startAngle, endAngle, WHITE, thickness)

cv2.imshow('fr', img)
cv2.waitKey(0)
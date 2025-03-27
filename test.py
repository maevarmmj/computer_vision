import cv2
import numpy as np

width, height = 900, 600
img = np.zeros((height, width, 3), np.uint8)
white_rect = np.array([[100, 100],
                       [width - 100, 100],
                       [width - 100, height - 100],
                       [100, height - 100],
                       [100, 100]], np.int32)

img = cv2.polylines(img, [white_rect], True, (255, 255, 255), 3)

line_start_x = int(white_rect[0:2, 0].mean())
line_start_y = int(white_rect[0:2, 1].mean())
line_end_x = int(white_rect[2:4, 0].mean())
line_end_y = int(white_rect[2:4, 1].mean())
clicked_x = 555
clicked_y = 192

img = cv2.circle(img, (clicked_x, clicked_y), 5, (0, 0, 255), -1)
img = cv2.circle(img, (line_start_x, line_start_y), 5, (0, 0, 255), -1)
img = cv2.circle(img, (line_end_x, line_end_y), 5, (0, 0, 255), -1)

cv2.imshow('fr', img)

pts = np.array([[line_start_x, line_start_y],
                [clicked_x, clicked_y],
                [line_end_x, line_end_y]], np.int32)

# side parabola coeffs

coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2)
poly = np.poly1d(coeffs)

yarr = np.arange(line_start_y, line_end_y)
xarr = poly(yarr)

parab_pts = np.array([xarr, yarr], dtype=np.int32).T
cv2.polylines(img, [parab_pts], False, (255, 0, 0), 3)

cv2.imshow('fr', img)
cv2.waitKey(0)
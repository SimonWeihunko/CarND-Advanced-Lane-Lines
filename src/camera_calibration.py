import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml
import os

os.chdir('..')  # move up one directory

# prepare object points
nx = 9  # enter the number of inside corners in x
ny = 6  # enter the number of inside corners in y

# initialize corresponding object and image points
objp_list = []
imgp_list = []

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

for num in range(2, 21):
    fname = 'camera_cal/calibration' + str(num) + '.jpg'
    img = cv2.imread(fname)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret is True:
        objp_list.append(objp)
        imgp_list.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        # f2, ax3 = plt.subplots(1, 1)
        # f2.tight_layout()
        # ax3.imshow(img)
        # plt.show()
        # cv2.imshow('img', img)
        # cv2.waitKey(100)


cv2.destroyAllWindows()

img = cv2.imread('camera_cal/calibration1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_list, imgp_list, gray.shape[::-1], None, None)

print(type(mtx), mtx.shape, mtx)
print(type(dist), dist.shape, dist)


for num in range(1, 21):
    # test the undistort result
    img = cv2.imread('camera_cal/calibration' + str(num) + '.jpg')
    h, w = img.shape[:2]
    dst = cv2.undistort(img, mtx, dist, None, None)

    f, (ax1, ax2) = plt.subplots(1, 2)
    f.tight_layout()
    ax2.imshow(dst)
    ax2.set_title('Undistort Image')
    ax1.imshow(img)
    ax1.set_title('Original Image')
    plt.show()

# save camera matrix and distort coefficient
with open('camera_mtx.yaml', 'w') as f:
    yaml.dump(mtx.tolist(), f)

with open('distort_coeff.yaml', 'w') as f:
    yaml.dump(dist.tolist(), f)

# verify yaml file
with open('distort_coeff.yaml') as f:
    loaded = yaml.load(f)
loaded = np.array(loaded)
print(type(loaded), loaded.shape, loaded)
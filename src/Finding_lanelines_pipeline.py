# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import yaml

os.listdir()  # move up one directory
os.chdir("..")  # move up one directory
image = mpimg.imread('test_images/straight_lines1.jpg')
os.chdir('src')
print('This image is:', type(image), 'with dimensions:', image.shape)

# load undistort parameters
with open('distort_coeff.yaml') as f:
    dist = yaml.load(f)
with open('camera_mtx.yaml') as f:
    mtx = yaml.load(f)
dist = np.array(dist)
mtx = np.array(mtx)

dst = cv2.undistort(image, mtx, dist, None, None)

# check undistort result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(dst)
# ax1.set_title('Undistort Image')
# ax2.imshow(image)
# ax2.set_title('Original Image')
# # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()


def color_and_gradient(img, s_thresh=(170, 255), sx_thresh=(30, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    sx_and_s_binary = sxbinary | s_binary
    return color_binary, sx_and_s_binary


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # TO-DO: Find the four below boundaries of the window #
        win_xleft_low = leftx_current - margin   # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high) & \
                         (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox <= win_xright_high) & \
                          (nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # TO-DO: If you found > minpix pixels, recenter next window #
        # (`right` or `leftx_current`) on their mean position #
        if len(good_left_inds) >= minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) >= minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 40 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Implement the calculation of R_curve (radius of curvature)
    left_curverad = (1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5 / abs(2*left_fit_cr[0])
    right_curverad = (1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5 / abs(2*right_fit_cr[0])

    return left_curverad, right_curverad


s_thresh = (160, 255)
sx_thresh = (30, 100)
result, sx_and_s_binary = color_and_gradient(dst, s_thresh, sx_thresh)

# Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
#
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=40)
#
# ax2.imshow(result)
# ax2.set_title('Color Thresholded Result', fontsize=40)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)



# warp the image
bottom_left_src = (229, 704)
bottom_right_src = (1095, 704)
top_left_src = (595, 450)
top_right_src = (690, 450)
bottom_left_dst = (290, 719)
bottom_right_dst = (980, 719)
top_left_dst = (290, 0)
top_right_dst = (980, 0)
src = np.float32([bottom_left_src, top_left_src, top_right_src, bottom_right_src])
dst = np.float32([bottom_left_dst, top_left_dst, top_right_dst, bottom_right_dst])


M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# save forward and inverse perspective transform
with open('perspective_transform.yaml', 'w') as f:
    yaml.dump(M.tolist(), f)
with open('inverse_perspective_transform.yaml', 'w') as f:
    yaml.dump(Minv.tolist(), f)


img_size = (sx_and_s_binary.shape[1], sx_and_s_binary.shape[0])

# Warp the image using OpenCV warpPerspective()
warped = cv2.warpPerspective(sx_and_s_binary, M, img_size, flags=cv2.INTER_LINEAR)


# f1, (ax3, ax4) = plt.subplots(1, 2, figsize=(24, 9))
# f1.tight_layout()
#
# ax3.imshow(sx_and_s_binary, cmap='gray')
# ax3.set_title('Original Image', fontsize=40)
# ax3.plot([bottom_left_src[0], top_left_src[0]], [bottom_left_src[1], top_left_src[1]], 'r-')
# ax3.plot([bottom_right_src[0], top_right_src[0]], [bottom_right_src[1], top_right_src[1]], 'r-')
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#
# ax4.imshow(warped, cmap='gray')
# ax4.plot([320, 320], [719, 0], 'r-')
# ax4.plot([950, 950], [719, 0], 'r-')
# ax4.set_title('Warped Result', fontsize=40)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = fit_polynomial(warped)

print(left_fitx.shape)
print(left_fitx[719])
print(right_fitx[719])

# Plots the left and right polynomials on the lane lines
f2, ax5 = plt.subplots()
ax5.imshow(out_img)
ax5.plot(left_fitx, ploty, color='yellow')
ax5.plot(right_fitx, ploty, color='yellow')

left_curverad, right_curverad = measure_curvature_real(ploty, left_fit, right_fit)
print(left_curverad, 'm', right_curverad, 'm')
plt.show()

# tracking
# Sanity Check
# Look ahead filter
# Smoothing
# Drawing
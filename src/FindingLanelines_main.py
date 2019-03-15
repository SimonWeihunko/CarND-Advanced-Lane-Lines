# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import yaml
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


def color_and_gradient(img, s_thresh=(170, 255), sx_thresh=(40, 100)):
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


def measure_curvature_real(ploty, left_fitx, right_fitx):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 40 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    ploty_m = ploty*ym_per_pix
    left_fitx_m = left_fitx*xm_per_pix
    right_fitx_m = right_fitx*xm_per_pix

    left_fit_m = np.polyfit(ploty_m, left_fitx_m, 2)
    right_fit_m = np.polyfit(ploty_m, right_fitx_m, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty_m)

    # Implement the calculation of R_curve (radius of curvature)
    left_curverad = (1 + (2*left_fit_m[0]*y_eval + left_fit_m[1])**2)**1.5 / abs(2*left_fit_m[0])
    right_curverad = (1 + (2*right_fit_m[0]*y_eval + right_fit_m[1])**2)**1.5 / abs(2*right_fit_m[0])

    return left_curverad, right_curverad


def lane_visualization(image, undist, warped, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result


def output_words_on_image(img, words):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (10, 100)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    cv2.putText(img, words,
                bottom_left_corner_of_text,
                font,
                font_scale,
                font_color,
                line_type)
    return img


# load camera matrix
with open('camera_mtx.yaml') as f:
    mtx = yaml.load(f)
mtx = np.array(mtx)
# load undistort parameters
with open('distort_coeff.yaml') as f:
    dist = yaml.load(f)
dist = np.array(dist)
# load perspective transform matrix
with open('perspective_transform.yaml') as f:
    M = yaml.load(f)
M = np.array(M)
# load inverse perspective transform matrix
with open('inverse_perspective_transform.yaml') as f:
    Minv = yaml.load(f)
Minv = np.array(Minv)


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')


class LaneDetection:

    def __init__(self, mtx, dist, M, Minv):
        self.left_line = Line()
        self.right_line = Line()
        self.mtx = mtx
        self.dist = dist
        self.M = M
        self.Minv = Minv

    def pipeline(self, img, s_thresh=(160, 255), sx_thresh=(30, 100)):

        undist = cv2.undistort(img, self.mtx, self.dist, None, None)

        result, sx_and_s_binary = color_and_gradient(undist, s_thresh, sx_thresh)

        # Warp the image using OpenCV warpPerspective()
        img_size = (sx_and_s_binary.shape[1], sx_and_s_binary.shape[0])
        warped = cv2.warpPerspective(sx_and_s_binary, M, img_size, flags=cv2.INTER_LINEAR)

        # find lane lines on the warped image
        out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = fit_polynomial(warped)

        new_detection_threshold = 0.2  # threshold to remove outliers
        # left lane tracking
        if self.left_line.detected:
            self.left_line.diffs = left_fit - self.left_line.current_fit

            if np.sum(np.abs(self.left_line.diffs/self.left_line.current_fit) < new_detection_threshold):
                self.left_line.current_fit = left_fit
                # append the most recent fit to the list
                if len(self.left_line.recent_xfitted) < 5:
                    self.left_line.recent_xfitted.append(left_fitx)
                else:
                    self.left_line.recent_xfitted.pop(0)
                    self.left_line.recent_xfitted.append(left_fitx)
            else:
                print(left_fit)
            # average x values of the fitted line over the last n iterations
            self.left_line.bestx = sum(self.left_line.recent_xfitted)/len(self.left_line.recent_xfitted)
            # print(self.left_line.bestx)
            # polynomial coefficients averaged over the last n iterations
            self.left_line.best_fit = np.polyfit(ploty, self.left_line.bestx, 2)
        else:
            self.left_line.detected = True
            # polynomial coefficients for the most recent fit
            self.left_line.current_fit = left_fit
            # x values of the last n fits of the line
            self.left_line.recent_xfitted = left_fitx
            # average x values of the fitted line over the last n iterations
            self.left_line.bestx = left_fitx
            # polynomial coefficients averaged over the last n iterations
            self.left_line.best_fit = left_fit
            # append the most recent fit to the list
            self.left_line.recent_xfitted = [left_fitx]

        # right lane tracking
        if self.right_line.detected:

            self.right_line.diffs = right_fit - self.right_line.current_fit

            if np.sum(np.abs(self.right_line.diffs / self.right_line.current_fit) < new_detection_threshold):
                self.right_line.current_fit = right_fit
                # append the most recent fit to the list
                if len(self.right_line.recent_xfitted) < 5:
                    self.right_line.recent_xfitted.append(right_fitx)
                else:
                    self.right_line.recent_xfitted.pop(0)
                    self.right_line.recent_xfitted.append(right_fitx)
            else:
                print(right_fit)
            # average x values of the fitted line over the last n iterations
            self.right_line.bestx = sum(self.right_line.recent_xfitted) / len(self.right_line.recent_xfitted)
            # print(self.right_line.bestx)
            # polynomial coefficients averaged over the last n iterations
            self.right_line.best_fit = np.polyfit(ploty, self.right_line.bestx, 2)
        else:
            self.right_line.detected = True
            # polynomial coefficients for the most recent fit
            self.right_line.current_fit = right_fit
            # x values of the last n fits of the line
            self.right_line.recent_xfitted = right_fitx
            # average x values of the fitted line over the last n iterations
            self.right_line.bestx = right_fitx
            # polynomial coefficients averaged over the last n iterations
            self.right_line.best_fit = right_fit
            # append the most recent fit to the list
            self.right_line.recent_xfitted = [right_fitx]

        # measure curvature
        left_curverad, right_curverad = measure_curvature_real(ploty, self.left_line.bestx, self.right_line.bestx)

        ###  lane visualization  ###
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_line.bestx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.bestx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result_img = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # calculate road curvature and veh position
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        road_curvature = (left_curverad + right_curverad) / 2
        veh_pos = (img.shape[1]//2 - 1 - (self.left_line.bestx[img.shape[0]-1] +
                                          self.right_line.bestx[img.shape[0]-1])
                   / 2) * xm_per_pix  # assume camera is at the center of vehicle

        # print curvature and vehicle position on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (10, 50)
        font_scale = 1
        font_color = (255, 255, 255)
        line_type = 2

        words_curvature = 'curvature of the road = ' + str(road_curvature) + ' m'
        words_veh_pos = 'vehicle position from lane center = ' + str(-veh_pos) + ' m'
        cv2.putText(result_img, words_curvature,
                    bottom_left_corner_of_text,
                    font,
                    font_scale,
                    font_color,
                    line_type)
        bottom_left_corner_of_text = (10, 100)
        cv2.putText(result_img, words_veh_pos,
                    bottom_left_corner_of_text,
                    font,
                    font_scale,
                    font_color,
                    line_type)

        return result_img


os.listdir()  # move up one directory
os.chdir("..")  # move up one directory
image = mpimg.imread('test_images/test2.jpg')
os.chdir('src')

lane_detection = LaneDetection(mtx, dist, M, Minv)
result = lane_detection.pipeline(image)

f2, ax5 = plt.subplots()
f2.tight_layout()
ax5.imshow(result)
ax5.set_title("final result")
plt.show()


os.chdir("..")  # move up one directory
name_of_video = 'project_video'
white_output = name_of_video + '_result.mp4'

clip1 = VideoFileClip(name_of_video + '.mp4')
white_clip = clip1.fl_image(lane_detection.pipeline)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

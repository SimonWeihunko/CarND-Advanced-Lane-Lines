## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image0]: ./output_images/chess_corners.png "chess corners"
[image1]: ./output_images/distort_and_undistort.png "Undistorted"
[image2]: ./output_images/undist_view.png "Road Transformed"
[image3]: ./output_images/color_thresholding.png "Binary Example"
[image4]: ./output_images/warp_result.png "Warp Example"
[image5]: ./output_images/lane_pixel_found.png "Fit Visual"
[image6]: ./output_images/final_result.png "Output"
[video1]: ./project_video_result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the python file located in `./src/camera_calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objp` is just a numpy array that contains coordinates (0,0,0), (1,0,0), (2,0,0),... to (6,5,0) and `objp_list` will 
be appended with a copy of it every time I successfully detect all chessboard corners in a test image. 
`imgp_list` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][image0]

I then used the output `objp_list` and `imgp_list` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of S channel and x derivative thresholds to generate a binary image (thresholding steps at lines #13 through #34 in `FindingLanelines_main.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appears in lines 220 through 246 in the file `FindingLanelines_demo_script.py`.  
I chose to hardcode the source and destination points in the following manner:

```python
# warp the image
bottom_left_src = (229, 704)
bottom_right_src = (1095, 704)
top_left_src = (595, 450)
top_right_src = (690, 450)
bottom_left_dst = (290-5, 719)
bottom_right_dst = (980-5, 719)
top_left_dst = (290-5, 0)
top_right_dst = (980-5, 0)
src = np.float32([bottom_left_src, top_left_src, top_right_src, bottom_right_src])
dst = np.float32([bottom_left_dst, top_left_dst, top_right_dst, bottom_right_dst])
```

This resulted in the following source and destination points:

| Source (x, y) | Destination (x, y)| 
|:-------------:|:-------------:| 
| 229, 704      | 285, 719      | 
| 595, 450      | 285, 0        |
| 1095, 704     | 975, 719      |
| 690, 450      | 975, 0        |

OpenCV provides `cv2.getPerspectiveTransform(src, dst)` and `cv2.warpPerspective(img, M, img_size)` to get the perspective transform matrix and warp the image to see the result.
I used the `./test_images/straight_lines1.jpg` to verify my perspective transform was working as expected. The lane lines in the warped image should look parallel.

 
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions `find_lane_pixels()` and `fit_polynomial()` take the warped image and return the
detected lane line pixels and second order fitted coefficients of lane lines. The `find_lane_pixels()` consists following steps:

1. Take a histogram of the bottom half of the image
2. Find the peak of the left and right halves of the histogram as the starting points of left and right lines
3. Set the window height and width
4. Set minimum number of pixels found to recenter window
5. Identify the x and y positions of all nonzero pixels within the windows
6. Update the window position if the pixels detected is greater than threshold
7. Repeat the process until hitting the boundary of image 

The `fit_polynomial()` simply takes the detected pixels and uses `np.polyfit()` function to find the best fitted coefficients. 
The following image shows the lane lines detection result

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The `measure_curvature_real()` function in `FindingLanelines_main.py` takes the polynomial coefficients and calculates the estimation of real curvature of the lines.
The formula is shown below.

```python
ym_per_pix = 40 / 720  # meters per pixel in y dimension
left_curverad = (1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5 / abs(2*left_fit_cr[0])
right_curverad = (1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5 / abs(2*right_fit_cr[0])
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The entire pipeline is inside the class `LaneDetection`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The entire lane detection pipeline is in the function `pipeline()` under the class `LaneDetection()`. 
In order to have a more robust detection, the class `line()` is added to track the detected lane lines, remove outliers 
and take an average of detected lines over the past n frames. After implementing this `line()` class, the lane detection algorithm 
can show stable result on the `project_video.mp4`. However, it fails catastrophically on the `challenge_video.mp4` and 
`harder_challenge_video.mp4`. In the `challenge_video.mp4`, the road is filled with irregularities and thus big gradient value
can appear on the undesired location, which in the end produces false positive detection. A potential solution is to add more 
color thresholding scheme like RGB threshold to detect white and yellow lines and filter out other high intensity region like shadow.
Another improvement we can make is to further limit the search region and make a more specific assumption of where the 
lane lines should be. Although this will limit the applicable scenarios of the algorithm, it will remove more false positive detections.
 
However, there are some cases the lane lines don't even exist, like vehicle making a big turn such that the left or right lines are out
of camera FOV. In the ultimate lane detection, we must come up a way to detect lanes not just with lines but with other 
information as well. 
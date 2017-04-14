**Vehicle Detection Project**

<span style="color:red"><b>Note, that to run the notebook, data must be downloaded as it is not included in the project repository.</b></span>

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image1a]: ./examples/alt_color_space_image.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The section **Data Exploration** is where the car / not-car data is read-in, specifically, in <span style="color:green"><b>Code Cell 25</b></span>, under the heading, **Extract the car / not-car image filenames into two lists.** The datasets are analyzed and compared to each other to judge their suitability in training in the same code cell.

Furthermore, in <span style="color:green"><b>Code Cell 27</b></span>, a function from <span style="color:green"><b>Code Cell 26</b></span>, `data_look()` is used to additionally, analyze the datasets.

In <span style="color:green"><b>Code Cell 28</b></span>, an example car and not-car image is displayed as repeated below:
![car/no-car image][image1]

While color space investigation can be found throughout the project, a substantial investigation is done between <span style="color:green"><b>Code Cell 10</b></span> and <span style="color:green"><b>Code Cell 17</b></span>. <span style="color:green"><b>Code Cell 31</b></span> holds the `extract_features_sbc_hist()` function, which returns color features. <span style="color:green"><b>Code Cell 20</b></span> offeres an example of an car image converted to another color space, as repeated below:
![alternate color space image][image1a]

<span style="color:green"><b>Code Cell 32</b></span> holds the `extract_features_hog()` function, which is an early investigation into HOG features and returns HOG features. Similar to color space investigation, HOG investigation is found thorughout the project, but a specific example is found at <span style="color:green"><b>Code Cell 30</b></span> and repeated below, using HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOG image][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and... used the parameters from the **Hog Sub-sampling Window Search** lesson. This was done to minimize errors in my learning (rather than network learning), particularly since much, much time was wasted in this part of the lessson to what I consider to be a calamitous mis-direction on Udacity's part. They reperesented an incorrect HOG extraction method in the lesson; one that couldn't possibly have been used to produce the results they presented. Debugging this consumed way too much time. So, with a node to expediency, I am trying to keep things simple until returning for an in-depth investigation. **Note:** please look into improving the modules beginning with module 30 and going forward, for future cohorts.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

<span style="color:green"><b>Code Cell 49</b></span> holds the `extract_features()` function, <span style="color:green"><b>Code Cell 51</b></span> is where the `analyze_and_output()` function is called. This function is in <span style="color:green"><b>Code Cell 34</b></span>. It trains a linear SVM using the HOG and color features extracted with `extract_features()`, outputs meta data, and returns `X_scaler` and `svc`, used by the `find_cars()` function in <span style="color:green"><b>Code Cell 61</b></span>.

<span style="color:green"><b>Code Cell 41</b></span> holds the results of a preliminary investigation into using both color and HOG features.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I did not spend much time on this part as I was aware that I would be implementing a more efficient algorithm later in the notebook.

In <span style="color:green"><b>Code Cell 43</b></span>, I played with an image and values to create windows that would result in positivie matches for cars.

<span style="color:green"><b>Code Cell 42</b></span> is the function, `slide_window()`, which produces the search, or, sliding windows for searching.

<span style="color:green"><b>Code Cell 44</b></span> is pretty much straight from the lesson, but using the values I came up with the sliding windows.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

The development for the pipeline was conducted in <span style="color:green"><b>Code Cell 68</b></span> through <span style="color:green"><b>Code Cell 74</b></span> leading to a final `process_frame()` function in <span style="color:green"><b>Code Cell 82</b></span>.
### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The processing has to be much faster than what it currently is. I'd add additional regions and scales in which to search for vehicles as I believe this would result in more accurate bounding boxes. I also believe that search windows may be absent at the horizontal edges, but this needs to be investigated further, before I am confident of this assertion.
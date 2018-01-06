# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform deep learning based detection algorithm to detect the cars on the road.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
---
### Neural network based cars detection algorithm

#### 1. Training images.

I started by reading in all the `vehicle` and `non-vehicle` images.

This wasn't enough as all of the car images were taken from the back and the model did not do well to detect the car untill the back was visible. So I augmented data from test video and generate more data of cars from the side along with thier back pictures.

#### 2. Explain how you you trained a classifier and settled on the current model.

Motivation for using neural network based solution was from [Max Ritter](https://github.com/maxritter/SDC-Vehicle-Lane-Detection), who has used neural work based solution for this project.

I started off with: 
* Scaled down version of YOLO
    - Using pretrained model and train the final layers on the current dataset. It did reasonably well, but was very slow.
* Tiny YOLO
    - Using pretrained weights but train the final layers again on the dataset.
    - Using pretrained weights.
    - Both did approximately similar to YOLO, still time consuming.
* Current model ([here](https://github.com/ShankHarinath/CarND-Vehicle-Detection/blob/master/model.py#L41)) similar to [Max Ritter](https://github.com/maxritter/SDC-Vehicle-Lane-Detection).
    - I tried different variants of it, added more layers (dropout, batch normalization), the barebone model seemed to work the best with some changes to the final layers.

I trainined the current model with the below network configuration and input dimension of (64, 64, 3) to create a binary classifier.

![TrainedModel](https://github.com/ShankHarinath/CarND-Vehicle-Detection/raw/master/output_images/TrainedModel.png)

I used the weights of this trained model and scaled the inputs to (260, 880, 3).
The models output is of size (None, 25, 103, 1), the 25*103 is the map of different regions on the image with bianry classification. This will tell us the hot area/possibility of finding a car. Detection model looks as below:

![FinalModel](https://github.com/ShankHarinath/CarND-Vehicle-Detection/raw/master/output_images/FinalModel.png)

Rescaled and cropped image from the video is then fed into the model to get the predictions.

![CroppedImage](https://github.com/ShankHarinath/CarND-Vehicle-Detection/raw/master/output_images/Cropped.png)

Prediction from the model

![PredictedImage](https://github.com/ShankHarinath/CarND-Vehicle-Detection/raw/master/output_images/Prediction.png)

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I saved the last 30 predictions data and used `cv2.groupRectangles()` to combine the bounding boxes predicted by the mdoel. On trial and error basis the parameters for the function is defined [here]().

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

* Detected boxes
![Detected boxes](https://github.com/ShankHarinath/CarND-Vehicle-Detection/raw/master/output_images/Boxes.png)
* `scipy.ndimage.measurements.label()` output
![GreyMap](https://github.com/ShankHarinath/CarND-Vehicle-Detection/raw/master/output_images/GreyMap.png)
* Heat map
![Heat Map](https://github.com/ShankHarinath/CarND-Vehicle-Detection/raw/master/output_images/HeatMap.png)
* Car Positions
![Heat & bounding boxes](https://github.com/ShankHarinath/CarND-Vehicle-Detection/raw/master/output_images/Car%20Positions.png)

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://github.com/ShankHarinath/CarND-Vehicle-Detection/raw/master/project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The neural network model fails to detect cars when it on the side as the dataset doesn't have too many images of cars from their side. I tried augmenting the data but it still can perform better with more images of different cars.

##### Enhancement:
* Can use `ImageDataGenerator` to augment images for brightness, rotation and sheer angle the images to get a better performance.


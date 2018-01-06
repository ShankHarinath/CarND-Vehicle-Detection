import glob
import pickle

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

class LaneDetector(): 
    def __init__(self, lane):
        self.lane = lane
        
    def detect_in_image(self, img, skip_history=False):
        img_copy = np.copy(img)
        # Undistort
        img_undistort = self.lane.image_pipeline._undistort_image(img_copy)

        # Perspective Transform
        img_unwarp = self.lane.image_pipeline._perspective_transform(img_undistort)

        # Sobel Absolute, Magnitude, Direction & color thresholds combined
        img_binary = self.lane.image_pipeline._combined_thresholds(img_unwarp)
        
        # Locate and draw lanes
        return self.lane.draw_lanes(img_binary, img_undistort, skip_history)
        
    def detect_in_video(self, video_path, output_path="project_video_result.mp4"):
        def gray(image):
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        clip = VideoFileClip(video_path)
        output_clip = clip.fl_image(self.detect_in_image)
        output_clip.write_videofile(output_path, audio=False)

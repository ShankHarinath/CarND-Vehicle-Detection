import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from collections import deque

from moviepy.editor import VideoFileClip
from lane_detector import LaneDetector
from scipy.ndimage.measurements import label


class DetectionPipeline:

    def __init__(self, model, lane_detector):
        self.lane_detector = lane_detector
        self.model = model
        self.history = deque(maxlen=30)

    def search_cars(self, detection_map):
        detection_map = np.squeeze(detection_map)
        (xx, yy) = np.meshgrid(np.arange(detection_map.shape[1]),
                               np.arange(detection_map.shape[0]))
        x = xx[detection_map > 0.8]
        y = yy[detection_map > 0.8]
        hot_windows = []

        # We save those rects in a list

        for (i, j) in zip(x, y):
            hot_windows.append(((i * 8 + 400, j * 8 + 400), (i * 8
                               + 464, j * 8 + 464)))

        return hot_windows

    def draw_boxes(
        self,
        img,
        bboxes,
        color=(0, 0, 255),
        thick=6,
        ):

        # Make a copy of the image

        draw_img = np.copy(img)

        # Iterate through the bounding boxes

        for bbox in bboxes:

            # Draw a rectangle given bbox coordinates

            cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)

        # Return the image copy with boxes drawn

        return draw_img

    def add_heat(self, heatmap, bbox_list):

        # Iterate through list of bboxes

        for box in bbox_list:

            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))

            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap

        return heatmap  # Iterate through list of bboxes

    def apply_threshold(self, heatmap, threshold):

        # Zero out pixels below the threshold

        heatmap[heatmap <= threshold] = 0

        # Return thresholded map

        return heatmap

    def draw_labeled_bboxes(
        self,
        img,
        labels,
        is_video=True,
        ):

        # Iterate through all detected cars

        for car_number in range(1, labels[1] + 1):

            # Find pixels with each car_number label value

            nonzero = (labels[0] == car_number).nonzero()

            # Identify x and y values of those pixels

            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            if not is_video:

                # Define a bounding box based on min/max x and y

                bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                        (np.max(nonzerox), np.max(nonzeroy)))

                # Draw the box on the image

                cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 6)
            else:

                # Append current boxes to history

                self.history.append([np.min(nonzerox),
                                    np.min(nonzeroy), np.max(nonzerox),
                                    np.max(nonzeroy)])

                # Get recent boxes for the last 30 fps

                recent_boxes = np.array(self.history).tolist()

                # Groups the object candidate rectangles with difference of 10%

                boxes = cv2.groupRectangles(recent_boxes, 5, .1)

                # Draw rectangles if found

                if len(boxes[0]) != 0:
                    for box in boxes[0]:
                        cv2.rectangle(img, (box[0], box[1]), (box[2],
                                box[3]), (0, 255, 0), 6)
        return img

    def find_cars(self, actual_image, is_video=True):
        crop = [(400, 660), (400, 1280)]

        if is_video:
            actual_image_rgb = actual_image
        else:
            actual_image_rgb = cv2.cvtColor(actual_image,
                    cv2.COLOR_BGR2RGB)

        image = actual_image_rgb[crop[0][0]:crop[0][1], crop[1][0]:
                                 crop[1][1], :]
        image = np.expand_dims(image, axis=0)

        detection_map = self.model.predict(image)

        self.hot_windows = self.search_cars(detection_map)
        self.all_boxes = self.draw_boxes(actual_image_rgb,
                self.hot_windows, (0, 255, 0), 6)

        heat = np.zeros_like(actual_image_rgb[:, :, 0]).astype(np.float)
        heat = self.add_heat(heat, self.hot_windows)

        # Apply threshold to help remove false positives

        heat = self.apply_threshold(heat, 3)

        # Visualize the heatmap when displaying

        self.heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function

        self.labels = label(self.heatmap)

        # Create the final image

        return self.draw_labeled_bboxes(np.copy(actual_image_rgb),
                self.labels, is_video)

    def combined_pipeline(self, img, is_video=True):
        image = np.copy(img)
        image = self.find_cars(image, is_video)

        # image = self.lane_detector.detect_in_image(image)

        return image

    def find_cars_in_video(self, video_path,
                           output_path='project_video_result.mp4'):
        clip = VideoFileClip(video_path)
        output_clip = clip.fl_image(self.combined_pipeline)
        output_clip.write_videofile(output_path, audio=False)


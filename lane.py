import numpy as np
import cv2

class Lane(object):
    def __init__(self, image_pipeline, left_lane, right_lane):
        self.image_pipeline = image_pipeline
        self.left_lane = left_lane
        self.right_lane = right_lane
        self.new_fit_count = 0
        self.prev_left_fit = None
        self.prev_right_fit = None
    
    def is_detected(self):
        return self.left_lane.detected & self.right_lane.detected
    
    def measure_curvature(self, img_warped, l_fit, r_fit, left_lane_inds, right_lane_inds):
        center_dist = 0
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension

        h = img_warped.shape[0]
        ploty = np.linspace(0, h-1, h)
        y_eval = np.max(ploty)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) != 0 and len(rightx) != 0:
            left_fit = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
            right_fit = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
            left_curverad = ((1 + (2*left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
            right_curverad = ((1 + (2*right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
            curverad = (left_curverad + right_curverad) / 2

        # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts 
        if r_fit is not None and l_fit is not None:
            car_position = img_warped.shape[1]/2
            l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
            r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
            lane_center_position = (r_fit_x_int + l_fit_x_int) /2
            center_dist = (car_position - lane_center_position) * xm_per_pix
        return curverad, center_dist
    
    def locate_lanes(self, warped_img):
        histogram = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        quarter_point = np.int(midpoint//2)
        leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
        rightx_base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint

        # Choose the number of sliding windows
        nwindows = 10
        # Set height of windows
        window_height = np.int(warped_img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 40
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        rectangles = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped_img.shape[0] - (window+1)*window_height
            win_y_high = warped_img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            rectangles.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        plot_data = (rectangles, histogram)
        
        return left_fit, right_fit, left_lane_inds, right_lane_inds, plot_data
        
    def locate_lanes_with_history(self, warped_img, left_fit_prev, right_fit_prev):
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 80
        left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) & 
                          (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) & 
                           (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) != 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) != 0:
            right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit, left_lane_inds, right_lane_inds
        
    def draw_lanes(self, warped_img, undistorted_img, skip_history=False):
        if not self.is_detected() or skip_history:
            left_fit, right_fit, left_lane_inds, right_lane_inds, _ = self.locate_lanes(warped_img)
            self.new_fit_count += 1
        else:
            left_fit, right_fit, left_lane_inds, right_lane_inds = self.locate_lanes_with_history(warped_img, self.left_lane.best_fit, self.right_lane.best_fit)
            self.new_fit_count = 0
            
        reset = self.new_fit_count >= 10
        self.left_lane.update_fit_for_new_frame(left_fit, reset)
        self.right_lane.update_fit_for_new_frame(right_fit, reset)
            
        curverad, center = self.measure_curvature(warped_img, self.left_lane.best_fit, self.right_lane.best_fit, left_lane_inds, right_lane_inds)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        roc = "Radius of curvature: {0:.2f}m".format(curverad)
        curv = "Distance from the Center: {0:.2f} m".format(center)
        
        ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
        left_fitx = self.left_lane.best_fit[0]*ploty**2 + self.left_lane.best_fit[1]*ploty + self.left_lane.best_fit[2]
        right_fitx = self.right_lane.best_fit[0]*ploty**2 + self.right_lane.best_fit[1]*ploty + self.right_lane.best_fit[2]
        
        warp_zero = np.zeros_like(warped_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.image_pipeline.Minv, (warped_img.shape[1], warped_img.shape[0])) 
        # Combine the result with the original image
        final_img = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
        cv2.putText(final_img, roc, (0, 50), font, 1.5, (255, 255, 255), 2)
        cv2.putText(final_img, curv, (0, 100), font, 1.5, (255, 255, 255), 2)
        
        if skip_history:
            self.left_lane.clear_all()
            self.right_lane.clear_all()
        return final_img

class Line(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False    
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
    def update_fit_for_new_frame(self, new_fit, reset=False):
        if reset:
            self.clear_all()
        if new_fit is not None:
            # Compare new and old fits
            if self.best_fit is not None:
                self.diffs = abs(new_fit - self.best_fit)
                
            if (self.diffs[0] > 0.001 or self.diffs[1] > 1.0 or self.diffs[2] > 100.) and len(self.current_fit) > 0:
                self.detected = False
            else:
                self.detected = True
                self.current_fit.append(new_fit)
                self.current_fit = self.current_fit[-10:]
                self.best_fit = np.mean(self.current_fit, axis=0)
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                self.current_fit = self.current_fit[1:]
                self.best_fit = np.mean(self.current_fit, axis=0)
    
    def clear_all(self):
        # was the line detected in the last iteration?
        self.detected = False    
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 


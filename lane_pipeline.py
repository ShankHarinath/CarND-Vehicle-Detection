import numpy as np
import cv2
from camera import Camera
import glob

class Pipeline(object):
    def __init__(self, camera):
        self.camera = camera

        # Source coords for perspective xform
        self.src = np.float32([[240, 719],
                                      [579, 450],
                                      [712, 450],
                                      [1165, 719]])
        # Dest coords for perspective xform
        self.dst = np.float32([[300, 719],
                                      [300, 0],
                                      [900, 0],
                                      [900, 719]])
        # Perspective Transform matrix
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        
        # Inverse Perspective Transform matrix
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
      
    def read_images(self, path):
        images = []
        for fname in glob.glob(path):
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        return images
    
    def _undistort_image(self, img):
        return cv2.undistort(img, self.camera.mtx, self.camera.dist, None, self.camera.mtx)
    
    def _perspective_transform(self, img):
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.M, (w, h), flags=cv2.INTER_LINEAR)
    
    def _sobel_abs_thresh(self, img, orient='x', thresh_min=25, thresh_max=255):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Return the result
        return binary_output
    
    def _sobel_mag_thresh(self, img, sobel_kernel=25, mag_thresh=(25, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output
    
    def _sobel_dir_thresh(self, img, sobel_kernel=7, thresh=(0, 0.09)):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output
    
    def _sobel_combined_thresh(self, img):
        gradx = self._sobel_abs_thresh(img, orient='x')
        grady = self._sobel_abs_thresh(img, orient='y')
        mag_binary = self._sobel_mag_thresh(img)
        dir_binary = self._sobel_dir_thresh(img)
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined
    
    def _color_thresh(self, img, thresh=(175, 250)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        # Use inRange instead of multiple thresholds
        s_thresh = cv2.inRange(s_channel.astype('uint8'), thresh[0], thresh[1])
        s_binary[(s_thresh == 255)] = 1
        return s_binary
        
    def _combined_thresholds(self, img):
        sobel_combined_thresh = self._sobel_combined_thresh(img)
        color_thresh = self._color_thresh(img)
        combined_binary = np.zeros_like(sobel_combined_thresh)
        combined_binary[(sobel_combined_thresh == 1) | (color_thresh == 1)] = 1
        return combined_binary
            
    def _color_gradient_threshold(self, img, s_thresholds=(175, 250), sx_thresholds=(30, 150)):
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        # Grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        retval, sxthresh = cv2.threshold(scaled_sobel, 30, 150, cv2.THRESH_BINARY)
        sxbinary[(sxthresh >= sx_thresholds[0]) & (sxthresh <= sx_thresholds[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        # Use inRange instead of multiple thresholds
        s_thresh = cv2.inRange(s_channel.astype('uint8'), s_thresholds[0], s_thresholds[1])
        s_binary[(s_thresh == 255)] = 1

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return combined_binary


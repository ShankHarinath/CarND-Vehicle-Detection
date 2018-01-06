import glob
import pickle

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

class Camera(object):
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        
        # Camera Matrix
        self.mtx = None
        # Camera Distortion coefficients
        self.dist = None
        # Camera translation vectors
        self.tvecs = None
        # Camera rotation vectors
        self.rvecs = None
    
    def calibrate_camera_from_file(self, path='calibration.p'):
        data = pickle.load(open( path, "rb" ))
        self.mtx = data['mtx']
        self.dist = data['dist']
        self.tvecs = data['tvecs']
        self.rvecs = data['rvecs']
        
    def save_calibration(self, path='calibration.p'):
        data = {
            'mtx': self.mtx,
            'dist': self.dist,
            'tvecs': self.tvecs,
            'rvecs': self.rvecs
        }
        pickle.dump(data, open( path, "wb" ))
        
    def calibrate_camera(self, images):
        objp = np.zeros((self.ny * self.nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []
        
        fig, axs = plt.subplots(5,4, figsize=(16, 11))
        fig.subplots_adjust(hspace = .2, wspace=.001)
        axs = axs.ravel()
        
        for i, img in enumerate(images):
            # Convert image to gray
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            # If corners are found, add object points, image points
            if ret is True:
                imgpoints.append(corners)
                objpoints.append(objp)
                
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            axs[i].axis('off')
            axs[i].imshow(img)
            
        # Calibrate camera
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

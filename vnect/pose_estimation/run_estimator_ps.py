#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import cv2
import time
import numpy as np
import multiprocessing as mp
from src import utils
from src.hog_box import HOGBox
from src.estimator import VNectEstimator
from arguments import parse_args
import os
from scipy.interpolate import interp1d

args = parse_args()

##################
### Parameters ###
##################
# the input camera serial number in the PC (int), or PATH to input video (str)
# video = 0
video = args.input
filename, _ = os.path.splitext(os.path.basename(video))
savepath = os.path.join(args.output_dir, filename)
# whether apply transposed matrix (when camera is flipped)
# T = True
T = False
# vnect input image size
box_size = 368
# parent joint indexes of each joint (for plotting the skeletal lines)
joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]


#######################
### Initializations ###
#######################
def init():
    # initialize VNect estimator
    global estimator
    estimator = VNectEstimator()

    # open a txt file to save angle data
    # global angles_file
    # angles_file = open('angles.txt', 'w')

    # catch the video stream
    global camera_capture
    camera_capture = cv2.VideoCapture(video)
    assert camera_capture.isOpened(), 'Video stream not opened: %s' % str(video)


################
### Box Loop ###
################
# use a simple HOG method to initialize bounding box
def hog_box():
    hog = HOGBox()
    ## click the mouse when ideal bounding box appears ##
    success, frame = camera_capture.read()
    # initialize bounding box as the maximum rectangle
    rect = [0, 0, frame.shape[1], frame.shape[0]]
#    while success and cv2.waitKey(1) == -1:
        # .copy() to prevent an unexpected bug
    frame = np.transpose(frame, axes=[1, 0, 2]).copy() if T else frame
#        choose, rect = hog(frame)
    rect = hog(frame)
#        if choose:
#            break
    success, frame = camera_capture.read()
    # the final static bounding box params
    return rect

# make 100 frame
def interpolation(data):
    data = np.array(data)
    n = data.shape[0]
    new_array = np.zeros(shape=(100, 21, 3))

    for i in range(21):
        skeleton_x = data[:, i, 0]
        skeleton_y = data[:, i, 1]
        skeleton_z = data[:, i, 2]

        x0 = np.linspace(0, 100, num=n, endpoint=True)
        y0 = skeleton_x
        f0 = interp1d(x0, y0, kind='quadratic')
        x1 = np.linspace(0, 100, num=n, endpoint=True)
        y1 = skeleton_y
        f1 = interp1d(x1, y1, kind='quadratic')
        x2 = np.linspace(0, 100, num=n, endpoint=True)
        y2 = skeleton_z
        f2 = interp1d(x2, y2, kind='quadratic')

        xnew = np.linspace(0, 100, num=100, endpoint=True)

        new_array[:, i, 0] = f0(xnew)
        new_array[:, i, 1] = f1(xnew)
        new_array[:, i, 2] = f2(xnew)

    return new_array

#################
### Main Loop ###
#################
## trigger any keyboard events to stop the loop ##
def main(q_start3d, q_joints):
    init()
    x, y, w, h = hog_box()
    q_start3d.put(True)
    t = time.time()
    success, frame = camera_capture.read()
    q_joints_list = []
   # width = int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
   # height = int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
   # fourcc = cv2.VideoWriter_fourcc(*'XVID')
   # writer = cv2.VideoWriter("output.avi", fourcc, 30.0,(width, height))
    while success and cv2.waitKey(1) == -1:
        # crop bounding box from the raw frame
        frame = np.transpose(frame, axes=[1, 0, 2]).copy() if T else frame
        frame_cropped = frame[y:y + h, x:x + w, :]
        # vnect estimating process
        joints_2d, joints_3d = estimator(frame_cropped)
        q_joints.put(joints_3d)
        q_joints_list.append(joints_3d)

        # 2d plotting
        frame_square = utils.img_scale_squarify(frame_cropped, box_size)
        frame_square = utils.draw_limbs_2d(frame_square, joints_2d, joint_parents)
        cv2.imshow('2D Prediction', frame_square)
    #    writer.write(frame_square)

        # write angle data into txt
        # angles_file.write('%f %f %f %f %f %f %f %f\n' % tuple([a for a in angles]))

        success, frame = camera_capture.read()

    q_joints_numpy = interpolation(q_joints_list)
    np.save(savepath, q_joints_numpy)
    # angles_file.close()
    try:
        camera_capture.release()
     #   writer.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        raise


if __name__ == '__main__':
    q_start3d = mp.Queue()
    q_joints = mp.Queue()
    ps_main = mp.Process(target=main, args=(q_start3d, q_joints))
    ps_plot3d = mp.Process(target=utils.plot_3d, args=(q_start3d, q_joints, joint_parents,savepath,args.savegif), daemon=True)

    ps_main.start()
    ps_plot3d.start()
    ps_main.join()

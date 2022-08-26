#!python3
import cv2 as cv
import numpy as np
import os
from time import time
# from windowcapture import WindowCapture
from vision import Vision
import pyscreenshot as ImageGrab
# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# initialize the WindowCapture class
#wincap = WindowCapture('/clover0/main_camera/image_raw')

# load the trained model
cascade_limestone = cv.CascadeClassifier('limestone_model_final.xml')
# load an empty Vision class
vision_limestone = Vision(None)

loop_time = time()
while(True):

    # get an updated image of the game
    screenshot = ImageGrab.grab()  # bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img_np = np.array(screenshot)  # this is the array obtained from conversion
    frame = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
    # do object detection
    rectangles = cascade_limestone.detectMultiScale(frame)

    # draw the detection results onto the original image
    detection_image = vision_limestone.draw_rectangles(frame, rectangles)

    # display the images
    cv.imshow('Matches', detection_image)

    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # press 'f' to save screenshot as a positive image, press 'd' to
    # save as a negative image.
    # waits 1 ms every loop to process key presses
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break
    elif key == ord('f'):
        cv.imwrite('positive/{}.jpg'.format(loop_time), frame)
    elif key == ord('d'):
        cv.imwrite('negative/{}.jpg'.format(loop_time), frame)

print('Done.')

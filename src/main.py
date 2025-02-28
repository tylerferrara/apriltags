from pathlib import Path
import cv2 as cv
import numpy as np
from glob import glob
import dt_apriltags as dt

# Fetch camera params from csv file
def get_camera_params():
    result = []
    src_dir = Path(__file__).resolve().parent
    with open(str(src_dir) + '/camera_params.csv', 'r') as f:
        for line in f:
            nums = line.split(',')
            for n in nums:
                result.append(float(n))
            break
    return result

# Returns a list of relative paths for every apriltag image 
def get_files_of_tags():
    root_proj_dir = Path(__file__).resolve().parent.parent
    return glob(str(root_proj_dir) + '/photos/tag36h11_200mm/*.jpg')

# Shows a bright green bounding box around the apriltag (if found)
def show_bounding_box(file, detector, camera_params, tag_size_meters):
    img = cv.imread(file, cv.IMREAD_GRAYSCALE)
    detections = detector.detect(img,
                                 estimate_tag_pose=True,
                                 camera_params=camera_params,
                                 tag_size=tag_size_meters)
    color = (0, 255, 0) # green
    thickness = 40 # in pixels
    for tag in detections:
        print("file: {} looks like tag id={}".format(file, tag.tag_id))
        # draw the 'bounding box' onto the image
        corners = tag.corners.astype(int) # opencv requires int points for drawing lines
        img = cv.imread(file) # overrite the img with original color
        img = cv.line(img, corners[0], corners[1], color, thickness)
        img = cv.line(img, corners[1], corners[2], color, thickness)
        img = cv.line(img, corners[2], corners[3], color, thickness)
        img = cv.line(img, corners[3], corners[0], color, thickness)
        img = cv.resize(img, (0, 0), fx = 0.1, fy = 0.1) # scale down the image
        cv.imshow("bounding box", img)
        cv.waitKey(0)
        return True
    return False


def main():
    # we are using tag36h11 with size 200mm
    tag_size_meters = 0.2
    # use the calibrate script for accurate camera params
    camera_params = get_camera_params()
    if (len(camera_params) == 0):
        print("ERROR: Failed to get camera params")
        return 1

    detector = dt.Detector(families='tag36h11', nthreads=8, debug=0)
    files = get_files_of_tags()

    for file in files:
        ret = show_bounding_box(file, detector, camera_params, tag_size_meters)
        if not ret:
            print("Unable to find corners in {}".format(file))
    

if __name__=="__main__":
    main()

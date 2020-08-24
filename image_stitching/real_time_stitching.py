import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import parameters as params
import utilities as utils
import homography


def main():
    # read images
    images = utils.read_images()
    # feature finding and image matching
    matched_homography = homography.find_homography(images)


if __name__ == '__main__':
    main()

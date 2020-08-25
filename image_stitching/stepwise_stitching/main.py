
'''
@brief: 
@author: Vijendra Singh
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import parameters as params
import utilities as utils
import core


def main():
    # read images
    images = utils.read_images()
    images_new = [images[0], images[1]]
    # feature finding and image matching
    h_mat = core.find_homography(images_new)
    # stitch image
    if h_mat is not None:
        core.stitch(images_new, h_mat[0][2], h_mat[0][3])
    # stitched = cv2.warpPerspective(
    #     images_new[0], h_mat[0][2], tuple(2*x for x in images_new[0].shape))
    # cv2.imshow('stitched', stitched)
    # cv2.moveWindow('stitched', 100, 50)
    # cv2.waitKey(0)
    # cv2.destroyWindow('stitched')


if __name__ == '__main__':
    main()


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
    # l2r
    # src_image = images[2][1]
    # dest_image = images[3][1]

    src_image = images[0][1]
    dest_image = images[1][1]

    # r2l
    # src_image = images[3][1]
    # dest_image = images[2][1]

    # t2b
    # src_image = images[0][1]
    # dest_image = images[2][1]
    # src_image = images[1][1]
    # dest_image = images[0][1]
    images_stitch = [src_image, dest_image]

    # feature finding and image matching
    h_mat = core.find_homography(images_stitch, 'image01')
    # stitch image
    if h_mat is not None:
        # core.stitch(images_stitch, h_mat[0][2], h_mat[0][3])
        image01 = core.warp(src_image, dest_image,  h_mat[0][2],
                            h_mat[0][3][0], h_mat[0][3][1], direction='l2r')
        cv2.imwrite('output/image01.png', image01)
    src_image = images[2][1]
    dest_image = images[3][1]
    images_stitch = [src_image, dest_image]
    # feature finding and image matching
    h_mat = core.find_homography(images_stitch, 'image23')
    # stitch image
    if h_mat is not None:
        # core.stitch(images_stitch, h_mat[0][2], h_mat[0][3])
        image23 = core.warp(src_image, dest_image,  h_mat[0][2],
                            h_mat[0][3][0], h_mat[0][3][1], direction='l2r')
        cv2.imwrite('output/image23.png', image23)
    src_image = image01
    dest_image = image23
    images_stitch = [src_image, dest_image]
    # feature finding and image matching
    h_mat = core.find_homography(images_stitch, 'image01_23')
    # stitch image
    if h_mat is not None:
        # core.stitch(images_stitch, h_mat[0][2], h_mat[0][3])
        image01_02 = core.warp(src_image, dest_image,  h_mat[0][2],
                               h_mat[0][3][0], h_mat[0][3][1],
                               direction='t2b', dest_overlay=True)
        cv2.imwrite('output/image01_23.png', image01_02)


if __name__ == '__main__':
    main()

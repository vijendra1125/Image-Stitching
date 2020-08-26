
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

    src_image = images[0]
    dest_image = images[1]

    # feature finding and image matching
    h_mat, bm = core.get_homography_matrix(src_image, dest_image, 'image01')
    # stitch image
    if h_mat is not None:
        image01 = core.warp(src_image, dest_image, h_mat,
                            bm[0], bm[1], direction='l2r')
        cv2.imwrite('output/image01.png', image01)

    src_image = images[2]
    dest_image = images[3]
    # feature finding and image matching
    h_mat, bm = core.get_homography_matrix(src_image, dest_image, 'image23')
    # stitch image
    if h_mat is not None:
        # core.stitch(images_stitch, h_mat[0][2], h_mat[0][3])
        image23 = core.warp(src_image, dest_image, h_mat,
                            bm[0], bm[1], direction='l2r')
        cv2.imwrite('output/image23.png', image23)

    src_image = image01
    dest_image = image23
    # feature finding and image matching
    h_mat, bm = core.get_homography_matrix(src_image, dest_image, 'image01_23')
    # stitch image
    if h_mat is not None:
        image01_23 = core.warp(src_image, dest_image, h_mat,
                               bm[0], bm[1], direction='t2b')
        cv2.imwrite('output/image01_23.png', image01_23)


if __name__ == '__main__':
    main()

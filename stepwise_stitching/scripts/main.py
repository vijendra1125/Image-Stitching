#
# Created on Thu Aug 24 2020
# Author: Vijendra Singh
# Licence: MIT
# Brief:
#


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

    # options to stitch upper half
    # image_01 = core.stitch(images[0], images[1], 'l2r', 'image_01')
    # image_10 = core.stitch(images[1], images[0], 'r2l', 'image_10')
    # options to stitch lower half
    # image_23 = core.stitch(images[2], images[3], 'l2r', 'image_23')
    # image_32 = core.stitch(images[3], images[2], 'r2l', 'image_32')

    # options to stitch left half
    image_02 = core.stitch(images[0], images[2], 't2b', 'image_02')
    # image_20 = core.stitch(images[2], images[0], 'b2t', 'image_20')
    # options to stitch right half
    image_13 = core.stitch(images[1], images[3], 't2b', 'image_13')
    # image_31 = core.stitch(images[3], images[1], 'b2t', 'image_31')

    # options to stich upper and lower half
    # image_01_23 = core.stitch(image_01, image_23, 't2b', 'full_image_01_23')
    # image_23_01 = core.stitch(image_23, image_01, 'b2t', 'full_image_23_01')
    # image_10_32 = core.stitch(image_10, image_32, 't2b', 'full_image_10_32')
    # image_32_10 = core.stitch(image_32, image_10, 'b2t', 'full_image_32_10')

    # options to stitch left half and right half
    image_02_13 = core.stitch(image_02, image_13, 'l2r', 'full_image_02_13')
    # image_13_02 = core.stitch(image_13, image_02, 'r2l', 'full_image_13_02')
    # image_20_31 = core.stitch(image_20, image_31, 'l2r', 'full_image_20_31')
    # image_31_20 = core.stitch(image_31, image_20, 'r2l', 'full_image_31_20')


if __name__ == '__main__':
    main()

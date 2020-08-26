
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
    image_01 = core.stitch(images[0], images[1], 'l2r', 'image_01')
    image_23 = core.stitch(images[2], images[3], 'l2r', 'image_23')
    image_all = core.stitch(image_01, image_23, 't2b', 'image_all')


if __name__ == '__main__':
    main()

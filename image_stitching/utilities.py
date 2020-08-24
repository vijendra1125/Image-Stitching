import os
import cv2

import parameters as params


def read_images(dir_name='data'):
    '''
    @brief: 
    @args[in]:
    @args[out]:
    '''
    images = []
    for image_name in os.listdir(dir_name):
        image = cv2.imread(os.path.join(dir_name, image_name), 0)
        width = image.shape[1]//params.DOWN_FACTOR
        height = image.shape[0]//params.DOWN_FACTOR
        image = cv2.resize(image, (width, height))
        images.append(image)
        if params.TEST_BOOL:
            cv2.imshow(image_name, image)
            cv2.waitKey(0)
    return images

import os
import cv2

# params
VIZ_BOOL = True
SCALE_FACTOR = 1/6
DATA_DIR = '../data'
OUTPUT_DIR = '../output'


def stitch_images(images):
    '''
    @brief: Stich all the given images and save the stitched image
    @arg[in]: Array of images to be stitched
    '''
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)
    if status == 0:
        out_path = os.path.join(OUTPUT_DIR, 'stitched.png')
        cv2.imwrite(out_path, stitched)
        if VIZ_BOOL:
            cv2.imshow("Stitched", stitched)
            cv2.waitKey(0)
    else:
        print("Stitching failed, retake images")


def main():
    images = []
    for image_name in os.listdir(DATA_DIR):
        image = cv2.imread(os.path.join(DATA_DIR, image_name))
        image = cv2.resize(image, dsize=(0, 0),
                           fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        if VIZ_BOOL:
            cv2.imshow(image_name, image)
            cv2.waitKey(0)
        images.append(image)
    stitch_images(images)


if __name__ == '__main__':
    main()

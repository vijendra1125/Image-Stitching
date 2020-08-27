# Image Stitching
## About
[Sticher class](https://docs.opencv.org/master/d2/d8d/classcv_1_1Stitcher.html) from OpenCV provides very good High-level API for stitching of images. It also provides option to use GPU (I have not tested it though) for making stitching process faster. Despite of all the convience provided by OpenCV Sticher class, very less control is provided by it whith [python binding](https://docs.opencv.org/4.1.2/d1/d46/group__stitching.html) whereas more control could be very useful in some applications. Think of all those applications of where images which need to be stitched are coming from cameras having fixed realtive positon. Such application could skip iterative claulation of homography matrix to reduce runtime drastically. Also control over algorithm used for feature extraction and feature matching could help performing better in some cases. Here we will be implementing stitching step-wise to learn the stitching proces along with getting more control over it even while using it in python. 

## Steps
Below are the main steps involved in the stitching of images:
* Feature extraction: SURF is better than SIFT. Since SIFT patent got expired in March 2020 whereas SUFR have ctive paptent. Program provides option to configure the algorithm you would like to use for extracting feature.
* Feature matching: FLANN is used for feature matching
* Find homography matrix 
* Warp the source image to destination using hompgraphy matrix and overlay destination image on top of it

## Dependencies
* Python 3.7.4
* OpenCV 3.4.2.16

## Good Links
* [Feature Descriptor matching using FLANN](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)

# Stepwise Image Stitching
## About
[Sticher class](https://docs.opencv.org/master/d2/d8d/classcv_1_1Stitcher.html) from OpenCV provides very good API for stitching of images. After using Stitcher class Higher Level API I got curious to know atleast basic principles behind stitching process and hence this subfolder appeared. Here I implemented a basic stitching pipeline with aim to develop understanding of stitching process. I hope I get time in future to make it more shophisticated. OpenCV also provides some detailed implementation of stitching using Stitcher class both in [C++](https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp) and [python](https://github.com/opencv/opencv/blob/master/samples/python/stitching_detailed.py). It also provides option to use GPU wheras I have not tested myself. 

## Steps
Below are the main steps involved in the stitching of images:
* Feature detection: In order to find homography matrix we need at least 4 matching feature points in two images. In this step we extract features from images. 
  * While implemeting extractor I have added SIFT and SURF only, I will motivate to try other algorithms too like ORB, BRISK, AKAZE. 
  * Among SURF and SIFT, SURF is better than SIFT. Since SIFT patent got expired in March 2020 whereas SURR have active patent. 
* Feature matching: At this step we find the matching features and filter out the featuers which doesnt have any match
  * While implementing matcher, Fast Library for Approximate Nearest Neighbour (FLANN) is used because it is faster than Brute Force Method
* Homography matrix: At this step we find homogarphy matrix as well as filter outliers 
* Warping: Warp the source image to destination using hompgraphy matrix, adjust it poision in warped output image and overlay destination image on top of it

## Dependencies
* Python 3.7.4
* OpenCV 3.4.2.16

## Demo
<p align="center">
<img src="./media/stepwise_stitching.gif" alt="drawing" width="512"/>
</p>

## Good Links
[TODO]

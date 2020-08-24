import cv2
import numpy as np
import sys

import parameters as params


def FLANN_matcher(kp1, des1, kp2, des2):
    '''
    @brief: match descriptor vectors with a FLANN based matcher
    @args[in]: 
        des1: descriptor from image 1
        des1: descriptor from image 2
    @args[out]: good matches
    '''
    # match descriptor vectors
    flann = cv2.FlannBasedMatcher(params.FLANN_INDEX_PARAMS,
                                  params.FLANN_SEARCH_PARAMS)
    matches = flann.knnMatch(des1, des2, k=params.FLANN_K)
    # find good matches as per Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
    if len(good_matches) < params.MIN_MATCH_COUNT:
        print("Not enough matches are found - {}/{}".format(len(good_matches),
                                                            params.MIN_MATCH_COUNT))
        return (None, None, None)
    # find kepoints for good matches
    good_kp1 = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    good_kp2 = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return (good_matches, good_kp1, good_kp2)


def matches_visualization(image1, image2, kp1, kp2, good_matches, h_mat, matches_mask):
    '''
    @brief: 
    @args[in]:
    @args[out]:
    '''
    h, w = image1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                      [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, h_mat)
    image2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matches_mask,  # draw only inliers
                       flags=2)
    image3 = cv2.drawMatches(image1, kp1, image2, kp2,
                             good_matches, None, **draw_params)
    cv2.imshow('matching feature visualization', image3)
    cv2.waitKey(0)


def get_homography_matrix(image1, image2):
    '''
    @brief: 
    @args[in]:
    @args[out]:
    '''
    # feature matching
    if params.FEATURE_MATCHING_ALGORITHM == 'sift':
        detector = cv2.xfeatures2d_SIFT.create()
    elif params.FEATURE_MATCHING_ALGORITHM == 'surf':
        detector = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
    else:
        print(
            '[ERROR]:please provide correct value to paramter FEATURE_MATCHING_ALGORITHM')
        sys.exit()
    kp1, des1 = detector.detectAndCompute(image1, None)
    kp2, des2 = detector.detectAndCompute(image2, None)
    good_matches, good_kp1, good_kp2 = FLANN_matcher(kp1, des1, kp2, des2)
    if good_matches is None:
        return None

    # homography matrix
    h_mat, mask = cv2.findHomography(good_kp1, good_kp2, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # matching feature vizualization
    if params.TEST_BOOL:
        matches_visualization(image1.copy(), image2.copy(),
                              kp1, kp2, good_matches, h_mat, matches_mask)

    return h_mat


def find_homography(images):
    '''
    @brief: 
    @args[in]:
    @args[out]:
    '''
    homography = []
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            if(i != j):
                image1 = images[i]
                image2 = images[j]
                h_mat = get_homography_matrix(image1, image2)
                if h_mat is not None:
                    homography.append([i, j, h_mat])
    return homography

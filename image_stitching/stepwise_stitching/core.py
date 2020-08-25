'''
@brief:
@author: Vijendra Singh
'''

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
    # find good matches as per Lowe's ratio test (0.7)
    # also find closest feature
    good_matches = []
    best_match = (0, 0)
    min_distance = 1
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
            if m.distance < min_distance:
                best_match = (kp1[m.queryIdx].pt, kp2[m.trainIdx].pt)
                min_distance = m.distance
    if len(good_matches) < params.MIN_MATCH_COUNT:
        print("Not enough matches are found - {}/{}".format(len(good_matches),
                                                            params.MIN_MATCH_COUNT))
        return (None, None, None, None)
    # find keypoints for good matches
    good_kp1 = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    good_kp2 = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return (good_matches, good_kp1, good_kp2, best_match)


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
    cv2.moveWindow('matching feature visualization', 100, 50)
    cv2.waitKey(0)
    cv2.imwrite('output/matching_{}.png'.format(params.TEST_ID), image3)
    cv2.destroyWindow('matching feature visualization')


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
    good_matches, good_kp1, good_kp2, bm = FLANN_matcher(kp1, des1, kp2, des2)
    if good_matches is None:
        return None

    # homography matrix
    h_mat, mask = cv2.findHomography(good_kp1, good_kp2, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # matching feature vizualization
    if params.TEST_BOOL:
        matches_visualization(image1.copy(), image2.copy(),
                              kp1, kp2, good_matches, h_mat, matches_mask)
    return (h_mat, bm)


def find_homography(images):
    '''
    @brief:
    @args[in]:
    @args[out]:
    '''
    homography = []
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            h_mat, bm = get_homography_matrix(images[i], images[j])
            if h_mat is not None:
                homography.append([i, j, h_mat, bm])
    if len(homography) == 0:
        return None
    return homography


def stitch(images, H, bm):
    bm_kp1 = bm[0]
    bm_kp2 = bm[1]
    bm_kp1_temp = np.dot(H, (bm_kp1 + (1,)))
    bm_kp1 = [x/bm_kp1_temp[2] for x in bm_kp1_temp]
    x_delta = (bm_kp2[0] + images[0].shape[0]) - bm_kp1[0]
    y_delta = bm_kp1[1] - bm_kp2[1]
    print(x_delta, y_delta)
    x_offset = x_delta
    y_offset = y_delta
    T = np.array([[1, 0, x_offset],
                  [0, 1, y_offset],
                  [0, 0, 1]])
    print(T)
    print(H)
    H = np.dot(T, H)
    print(H)

    stitched_frame_size = tuple(2*x for x in images[0].shape)
    stitched = cv2.warpPerspective(images[0], H, stitched_frame_size)
    # stitched[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]
    stitched[0:images[0].shape[0],
             images[0].shape[1]:] = images[1]
    if params.TEST_BOOL:
        cv2.imshow('stitched', stitched)
        cv2.moveWindow('stitched', 100, 50)
        cv2.waitKey(0)
        cv2.imwrite('output/stitched_{}.png'.format(params.TEST_ID), stitched)
        cv2.destroyWindow('stitched')

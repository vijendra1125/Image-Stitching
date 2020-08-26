'''
@brief:
@author: Vijendra Singh
'''

import cv2
import numpy as np
import sys

import parameters as params


def FLANN_matcher(src_kp, dest_kp, src_desc, dest_desc):
    '''
    @brief: match descriptor vectors with a FLANN based matcher
    @args[in]:
        src_kp: keypoints from source image
        dest_kp: keyppoints from destination image 
        src_desc: descriptor from source image
        dest_desc: descriptor from destination image
    @args[out]: 
        good_mathes: matches with descriptor distance below Lowe's ration
        best_match: keypoints in soure  and destination image corresponding to 
                    best matches descriptor  
    '''
    # match descriptor vectors
    flann = cv2.FlannBasedMatcher(params.FLANN_INDEX_PARAMS,
                                  params.FLANN_SEARCH_PARAMS)
    matches = flann.knnMatch(src_desc, dest_desc, k=params.FLANN_K)

    # find good matches as per Lowe's ratio test (0.7)
    # also find keypoints in soure  and destination image corresponding to
    # best matches descriptor
    good_matches = []
    best_match = (0, 0)
    min_distance = 1
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
            if m.distance < min_distance:
                best_match = (src_kp[m.queryIdx].pt, dest_kp[m.trainIdx].pt)
                min_distance = m.distance
    if len(good_matches) < params.MIN_MATCH_COUNT:
        print("Not enough matches are found - {}/{}".format
              (len(good_matches), params.MIN_MATCH_COUNT))
        return (None, None, None, None)
    # find keypoints for good matches
    good_src_kp = np.float32(
        [src_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    good_dest_kp = np.float32(
        [dest_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return (good_matches, best_match, good_src_kp, good_dest_kp)


def matches_visualization(src_image, dest_image, src_kp, dest_kp, good_matches,
                          h_mat, matches_mask, task_name):
    '''
    @brief: visualize the good matches
    @args[in]:
        src_image: source image
        dest_image: destination image
        src_kp: keypoints from source image
        dest_kp: keypoints from destination image
        good_matches: matches with descriptor distance below Lowe's ration
        h_mat: homography matrix
        matches_mask: homography mask
    '''
    h, w = src_image.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                      [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, h_mat)
    dest_image = cv2.polylines(dest_image,
                               [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matches_mask,  # draw only inliers
                       flags=2)
    image3 = cv2.drawMatches(src_image, src_kp, dest_image, dest_kp,
                             good_matches, None, **draw_params)
    cv2.imshow('matching feature visualization', image3)
    cv2.moveWindow('matching feature visualization', 100, 50)
    cv2.waitKey(0)
    cv2.imwrite('output/matching_{}.png'.format(task_name), image3)
    cv2.destroyWindow('matching feature visualization')


def get_homography_matrix(src_image, dest_image, task_name):
    '''
    @brief: find homography matrix and keypoint in source and destination image 
            for best match
    @args[in]:
        src_image: source image
        dest_image: destination image
    @args[out]:
        h_mat: homography matrix
        best match: keypoint in source and destination image for best match
    '''
    # feature matching
    if params.FEATURE_MATCHING_ALGORITHM == 'sift':
        detector = cv2.xfeatures2d_SIFT.create()
    elif params.FEATURE_MATCHING_ALGORITHM == 'surf':
        detector = cv2.xfeatures2d_SURF.create(hessianThreshold=400)
    else:
        print('[ERROR]: Please provide correct value to paramter \
              FEATURE_MATCHING_ALGORITHM')
        sys.exit()
    src_kp, src_desc = detector.detectAndCompute(src_image, None)
    dest_kp, dest_desc = detector.detectAndCompute(dest_image, None)
    matcher_out = FLANN_matcher(src_kp, dest_kp, src_desc, dest_desc)
    good_matches = matcher_out[0]
    best_match = matcher_out[1]
    good_src_kp = matcher_out[2]
    good_dest_kp = matcher_out[3]
    if good_matches is None:
        return (None, None)

    # homography matrix
    h_mat, mask = cv2.findHomography(good_src_kp, good_dest_kp,
                                     cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # matching feature vizualization
    if params.TEST_BOOL:
        matches_visualization(src_image.copy(), dest_image.copy(),
                              src_kp, dest_kp, good_matches,
                              h_mat, matches_mask, task_name)
    return (h_mat, best_match)


def find_homography(images, task_name):
    '''
    @brief:
    @args[in]:
    @args[out]:
    '''
    homography = []
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            h_mat, bm = get_homography_matrix(images[i], images[j], task_name)
            if h_mat is not None:
                homography.append([i, j, h_mat, bm])
    if len(homography) == 0:
        return None
    return homography


def warp(src_image, dest_image,  H, src_bm_kp, dest_bm_kp, direction='hor', dest_overlay=True):
    # find the coordinate of best match keypoint from source image in the warped source image
    warped_src_bm_kp_t = np.dot(H, (src_bm_kp + (1,)))
    warped_src_bm_kp = [x/warped_src_bm_kp_t[2] for x in warped_src_bm_kp_t]
    # calculate offset based on the warping direction
    if direction == 'r2l':
        stitched_frame_size = (2 * src_image.shape[1], src_image.shape[0])
        x_offset = dest_bm_kp[0] - warped_src_bm_kp[0]
        y_offset = dest_bm_kp[1] - warped_src_bm_kp[1]
    elif direction == 'l2r':
        stitched_frame_size = (2*src_image.shape[1], src_image.shape[0])
        x_offset = dest_bm_kp[0]+src_image.shape[0] - warped_src_bm_kp[0]
        y_offset = dest_bm_kp[1] - warped_src_bm_kp[1]
    elif direction == 't2b':
        stitched_frame_size = (src_image.shape[1], 2*src_image.shape[0])
        x_offset = dest_bm_kp[0] - warped_src_bm_kp[0]
        y_offset = dest_bm_kp[1]+src_image.shape[1] - warped_src_bm_kp[1]
    # caculate new homography matrix with offset compensation
    T = np.array([[1, 0, x_offset],
                  [0, 1, y_offset],
                  [0, 0, 1]])
    print(T)
    print(H)
    H = np.dot(T, H)
    print(H)
    # warp the source image
    print(dest_image.shape, stitched_frame_size)
    stitched = cv2.warpPerspective(src_image, H, stitched_frame_size)
    # overlay destination image on warped output
    if dest_overlay:
        if direction == 'r2l':
            stitched[0:dest_image.shape[0], 0:dest_image.shape[1]] = dest_image
        elif direction == 'l2r':
            stitched[0:dest_image.shape[0], dest_image.shape[1]:] = dest_image
        if direction == 't2b':
            # stitched[dest_image.shape[0]:, 0:dest_image.shape[1]] = dest_image
            for i in range(dest_image.shape[0], dest_image.shape[0]*2):
                for j in range(dest_image.shape[1]):
                    if stitched[i, j] == 0:
                        stitched[i, j] = dest_image[i - dest_image.shape[0], j]
    # resize the final stitiched image and show
    stitched = cv2.resize(stitched, (504, 504))
    if params.TEST_BOOL:
        cv2.imshow('stitched', stitched)
        cv2.moveWindow('stitched', 100, 50)
        cv2.waitKey(0)
        cv2.destroyWindow('stitched')
    return stitched

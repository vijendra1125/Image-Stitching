#
# Created on Thu Aug 24 2020
# Author: Vijendra Singh
# Licence: MIT
# Brief:
#

import cv2
import numpy as np
import sys
import os

import parameters as params


def FLANN_matcher(src_desc, dest_desc):
    '''
    @brief: match descriptor vectors with a FLANN based matcher
    @args[in]:
        src_desc: descriptor from source image
        dest_desc: descriptor from destination image
    @args[out]:
        good_mathes: matches with descriptor distance below Lowe's ration
    '''
    # match descriptor vectors
    flann = cv2.FlannBasedMatcher(params.FLANN_INDEX_PARAMS,
                                  params.FLANN_SEARCH_PARAMS)
    matches = flann.knnMatch(src_desc, dest_desc, k=params.FLANN_K)

    # find good matches as per Lowe's ratio test (0.7)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
    if len(good_matches) < params.MIN_MATCH_COUNT:
        print("Not enough matches are found - {}/{}".format
              (len(good_matches), params.MIN_MATCH_COUNT))
        return (None, None, None, None)
    return good_matches


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
    out_path = os.path.join(params.OUTPUT_DIR, 'matching_'+task_name+'.png')
    cv2.imwrite(out_path, image3)
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
    good_matches = FLANN_matcher(src_desc, dest_desc)
    # find keypoints for good matches
    good_src_kp = np.float32(
        [src_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    good_dest_kp = np.float32(
        [dest_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    if good_matches is None:
        return (None, None)

    # homography matrix
    h_mat, mask = cv2.findHomography(good_src_kp, good_dest_kp,
                                     cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # find best match
    id = matches_mask.index(1)
    best_match = (tuple(good_src_kp[id][0]), tuple(good_dest_kp[id][0]))
    min_dist = good_matches[id].distance
    for idx, good_match in enumerate(good_matches):
        if (good_match.distance < min_dist) and matches_mask[idx] == 1:
            best_match = (tuple(good_src_kp[idx][0]),
                          tuple(good_dest_kp[idx][0]))
            min_dist = good_match.distance

    # matching feature vizualization
    if params.TEST_BOOL:
        matches_visualization(src_image.copy(), dest_image.copy(),
                              src_kp, dest_kp, good_matches,
                              h_mat, matches_mask, task_name)

    return (h_mat, best_match)


def warp(src_image, dest_image,  H, src_bm_kp, dest_bm_kp, direction, task_name):
    '''
    @brief:
    @args[in]:
    @args[out]:
    '''
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
        x_offset = dest_bm_kp[0] + src_image.shape[1] - warped_src_bm_kp[0]
        y_offset = dest_bm_kp[1] - warped_src_bm_kp[1]
    elif direction == 't2b':
        stitched_frame_size = (src_image.shape[1], 2*src_image.shape[0])
        x_offset = dest_bm_kp[0] - warped_src_bm_kp[0]
        y_offset = dest_bm_kp[1] + src_image.shape[0] - warped_src_bm_kp[1]
    elif direction == 'b2t':
        stitched_frame_size = (src_image.shape[1], 2*src_image.shape[0])
        x_offset = dest_bm_kp[0] - warped_src_bm_kp[0]
        y_offset = dest_bm_kp[1] - warped_src_bm_kp[1]
    else:
        print('[ERROR]: Didnt find correct direction for warping task with \
        task name {}'.format(task_name))
        sys.exit()

    # caculate new homography matrix with offset compensation
    T = np.array([[1, 0, x_offset],
                  [0, 1, y_offset],
                  [0, 0, 1]])
    H = np.dot(T, H)

    # warp the source image
    stitched = cv2.warpPerspective(src_image, H, stitched_frame_size)
    # overlay destination image on warped output
    if True:
        if direction == 'r2l':
            # stitched[0:dest_image.shape[0], 0:dest_image.shape[1]] = dest_image
            for i in range(0, dest_image.shape[0]):
                for j in range(0, dest_image.shape[1]):
                    if stitched[i, j] == 0:
                        stitched[i, j] = dest_image[i, j]
        elif direction == 'l2r':
            # stitched[0:dest_image.shape[0], dest_image.shape[1]:] = dest_image
            for i in range(0, dest_image.shape[0]):
                for j in range(dest_image.shape[1], dest_image.shape[1]*2):
                    if stitched[i, j] == 0:
                        stitched[i, j] = dest_image[i, j - dest_image.shape[1]]
        elif direction == 't2b':
            # stitched[dest_image.shape[0]:, 0:dest_image.shape[1]] = dest_image
            for i in range(dest_image.shape[0], dest_image.shape[0]*2):
                for j in range(dest_image.shape[1]):
                    if stitched[i, j] == 0:
                        stitched[i, j] = dest_image[i - dest_image.shape[0], j]
            stitched = cv2.resize(stitched, dsize=(0, 0), fx=1/2, fy=1/2)
        elif direction == 'b2t':
            # stitched[0:dest_image.shape[0], 0:dest_image.shape[1]] = dest_image
            for i in range(0, dest_image.shape[0]):
                for j in range(dest_image.shape[1]):
                    if stitched[i, j] == 0:
                        stitched[i, j] = dest_image[i, j]
            stitched = cv2.resize(stitched, dsize=(0, 0), fx=1/2, fy=1/2)

    if params.TEST_BOOL:
        cv2.imshow('stitched', stitched)
        cv2.moveWindow('stitched', 100, 50)
        cv2.waitKey(0)
        cv2.destroyWindow('stitched')
    return stitched


def stitch(src_image, dest_image, d, task_name):
    # feature finding and image matching
    h_mat, bm = get_homography_matrix(src_image, dest_image, task_name)
    # stitch image
    if h_mat is not None:
        image = warp(src_image, dest_image, h_mat,
                     bm[0], bm[1], d, task_name)
        out_path = os.path.join(params.OUTPUT_DIR, task_name+'.png')
        cv2.imwrite(out_path, image)
    return image

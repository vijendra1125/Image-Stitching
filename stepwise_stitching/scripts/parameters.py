#
# Created on Thu Aug 24 2020
# Author: Vijendra Singh
# Licence: MIT
# Brief:
#

# if true, enable additional print and visulaization
TEST_BOOL = True
# factor by image should be downscaled before use
DOWN_FACTOR = 6
# directory to read data from
DATA_DIR = '../data'
# directory to write ouput in
OUTPUT_DIR = '../output'


# ---Feature matching---
# feature matching algorithm, choose between 'sift' and 'surf'
FEATURE_MATCHING_ALGORITHM = 'sift'
# -matching-
# FLANN matcher parameters
FLANN_INDEX_PARAMS = dict(algorithm=1, trees=5)
FLANN_SEARCH_PARAMS = dict(checks=50)
FLANN_K = 2
# minimum number of feature matching in two image
MIN_MATCH_COUNT = 4

import cv2
import cv2.xfeatures2d
import numpy as np
import itertools
from asift import affine_detect, init_feature, affine_detect, filter_matches
from multiprocessing.pool import ThreadPool
from common import Timer

class Image:
    def __init__(self, filename):
        self.filename = filename
        self.data = cv2.imread(filename)
        self.kp = []
        self.kpdes = []


if __name__ == "__main__":
    # Load images
    images = [
        Image('res/pano1tenpercent.jpg'),
        Image('res/pano2tenpercent.jpg'),
        Image('res/pano3tenpercent.jpg')
    ]
    #Once images are loaded calc ASIFT of each one
    detector, matcher = init_feature('brisk-flann')
    pool=ThreadPool(processes = cv2.getNumberOfCPUs())
    for image in images:
        image.kp, image.kpdes = affine_detect(detector, image.data, pool=pool)
        imageMarkers = image.data.copy()
        for marker in image.kp:
            imageMarkers = cv2.drawMarker(
                imageMarkers,
                tuple(int(i) for i in marker.pt),
                color=(0,255,0)
            )
        # cv2.imshow('Image with markers: ' + image.filename,imageMarkers)
        # cv2.waitKey(0)
    # Once all features are obtained then match images for each pair
    n = 0
    for pair in itertools.combinations(images,2):
        with Timer('matching'):
            raw_matches = matcher.knnMatch(pair[0].kpdes, trainDescriptors = pair[1].kpdes, k = 2)
        p1, p2, kp_pairs = filter_matches(pair[0].kp, pair[1].kp, raw_matches)
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv.RANSAC, 5.0)
            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
            # do not draw outliers (there will be a lot of them)
            kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        else:
            H, status = None, None
            print('%d matches found, not enough for homography estimation' % len(p1))
        matches = sorted(matches, key = lambda x: x.distance)
        imMatches = cv2.drawMatches(
            pair[0].data, pair[0].kp,
            pair[1].data, pair[1].kp,
            matches[:10],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite('output' + str(n) + '.jpg',
            imMatches)
        n=n+1
        continue
    pass


import cv2
import cv2.xfeatures2d
import numpy as np
import itertools

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
    #Once images are loaded calc SIFT of each one
    sift1 = cv2.xfeatures2d.SIFT_create(
        nfeatures=2000,
        nOctaveLayers=3,
        contrastThreshold=0.05,
        edgeThreshold=6,
        sigma=1.6
    )
    mask = np.ones(shape=np.shape(images[0].data)[0:2],dtype=np.uint8)
    for image in images:
        image.kp, image.kpdes = sift1.detectAndCompute(image.data, mask)
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
    FLANN_INDEX_KDTREE = 0
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50))
    n = 0
    for pair in itertools.combinations(images,2):
        matches = matcher.match(
            pair[0].kpdes,
            pair[1].kpdes)
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


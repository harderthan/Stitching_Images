import cv2
import numpy as np

img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = None

matches = flann.knnMatch(des1, des2, k=2)

good = []

matches = sorted(matches, key=lambda x: x[0].distance)

for m, n in matches:
	if m.distance < 0.7 * n.distance:
		good.append(m)

if len(good) > 10:
	src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
	dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	matchesMask = mask.ravel().tolist()

	h,w,c = img1.shape
	pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts, M)
else:
	print("not enough matches")
	matchesMask = None

h1,w1 = img1.shape[:2]
h2,w2 = img2.shape[:2]
pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
pts1_ = cv2.perspectiveTransform(pts1, M)
pts = np.concatenate((pts1_, pts2), axis=0)
[xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
t = [-xmin,-ymin]
Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
result = cv2.warpPerspective(img1, Ht.dot(M), (xmax-xmin, ymax-ymin))
result[t[1]:h1+t[1],t[0]:w1+t[0]] = img2

cv2.imshow("result", result)
cv2.waitKey()
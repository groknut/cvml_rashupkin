
import cv2
import numpy as np

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0+cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, -6)

def match(gray, template, min_matches=10, thresh=0.75, proj_thresh=5.0):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < n.distance * thresh:
            good.append(m)

    if len(good) < 4:
        return None, 0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, proj_thresh)
    if H is None:
        return None, 0

    if mask.sum() < min_matches:
        return None, 0

    h, w = template.shape[:2]
    corners = np.float32([[0, 0], [0, h-1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(corners, H)
    return transformed, mask.sum()

template = None
while capture.isOpened():
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        break


    key = chr(cv2.waitKey(1) & 0xFF)
    if key == 'q':
        break
    elif key == 't':
        roi = cv2.selectROI("ROI", gray)
        cv2.destroyWindow("ROI")
        template = gray[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    if template is not None:
        bbox, count = match(gray=gray, template=template, min_matches=5)
        if bbox is not None:
            cv2.polylines(frame, [np.int32(bbox)], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("Template", template)
    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()

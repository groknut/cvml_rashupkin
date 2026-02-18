import cv2
import numpy as np
import uuid
from pathlib import Path

"""
====== SETTINGS ======

There is settings for camera
"""

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0+cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
capture.set(cv2.CAP_PROP_EXPOSURE, -3)

"""
There is settings for paint
"""

settings = {
    "ball": {
        "color": None
    },
    "path": []
}

saved_path = Path(__file__).parent / "saved"

if not saved_path.exists():
    saved_path.mkdir()

"""
====== UTILS ======

There is functions for calculate color and get ball from frame
"""

def get_color(image):
    x, y, w, h = cv2.selectROI("ColorSelection", image)
    roi = image[y:y+h, x:x+w]
    color = (np.median(roi[:, :, 0]), np.median(roi[:, :, 1]), np.median(roi[:, :, 2]))
    cv2.destroyWindow("ColorSelection")
    return color

def get_ball(image, color):
    lower = (np.max(color[0] - 5, 0), color[1] * 0.8 ,color[2]*0.8)
    upper = (color[0] + 5, 255, 255)
    mask = cv2.inRange(image, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        x, y, radius = int(x), int(y), int(radius)
        return True, (x, y, radius, mask)
    return False, (-1, -1, -1, np.array(([])))

def find_center(frame, mask, x, y, color):
    M = cv2.moments(mask)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else: 
        cx, cy = x, y
    frame = cv2.circle(frame, (cx, cy), 10, color, 3)
    return (cx, cy)
    
"""
====== MAIN ======

The main part of the script. 

CONTROL:
`q`: exit
`a`: detect ball
`c`: clear frame
`s`: save image
"""

while capture.isOpened():
    ret, frame = capture.read()    
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    key = chr(cv2.waitKey(1) & 0xFF)

    if key == 'q':
        break

    elif key == 'a':
        color = get_color(hsv)
        settings["ball"]["color"] = color
        settings["ball"]["color"] = color
        print(settings["ball"]["color"])

    elif key == 'c':
        settings["path"] = []

    elif key == 's' and len(settings["path"]) > 2:
        path = np.array(settings["path"], dtype=np.int32)
        shape = (max(path[:,1])+1, max(path[:, 0])+1)
        saved = np.zeros(shape, dtype=np.uint8)
        for i in range(0, len(path)-1, 1):
            cv2.line(saved, path[i][:2], path[i+1][:2], color=255, thickness=path[i][-1])
        filename = saved_path / f"image_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(filename, saved)
        print("Изображение было сохранено: {}".format(filename))

    if settings["ball"]["color"]:
        
        retr, (x, y, radius, mask) = get_ball(hsv, settings["ball"]["color"])

        if retr:

            nx, ny = find_center(frame, mask, x, y, settings["ball"]["color"])
            settings["path"].append([nx, ny, max(1, int(radius / 10))])
            path = np.array(settings["path"], dtype=np.int32)
            
            if len(path) > 2:
                for i in range(0, len(path)-1, 1):
                    cv2.line(frame, path[i][:2], path[i+1][:2], color=settings["ball"]["color"], thickness=path[i][-1])

            cv2.imshow("Mask", mask)
            cv2.circle(frame, (x, y), radius, (4, 135, 80))

        else:
            path = np.array(settings["path"], dtype=np.int32)
            
            if len(path) > 2:
                for i in range(0, len(path)-1, 1):
                    cv2.line(frame, path[i][:2], path[i+1][:2], color=settings["ball"]["color"], thickness=path[i][-1])
    
    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()


import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound
import numpy as np
from pathlib import Path

# вспомогательные функции
# для вычисления локтевого угла (до 180 градусов)
def get_angle(a, b, c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0])
    ab = np.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle

# вычисление среднего угла сгибания рук
def get_avg_angle(keypoints):
    left_angle = get_angle(keypoints[0][5], keypoints[0][7], keypoints[0][9])
    right_angle = get_angle(keypoints[0][6], keypoints[0][8], keypoints[0][10])
    return (left_angle + right_angle) / 2

# проверить, отжался ли человек (левый и правый локтевые углы >= 160, практически выпрямлены)
def pushup(keypoints):
    nose_seen = keypoints[0][0] > 0 and keypoints[0][1] > 0
    eyes_seen = (
        keypoints[1][0] > 0
        and keypoints[1][1] > 0
        and keypoints[2][0] > 0
        and keypoints[2][1] > 0
    )

    if not (nose_seen and eyes_seen): return None

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    left_angle = get_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = get_angle(right_shoulder, right_elbow, right_wrist)

    return left_angle >= 160 and right_angle >= 160

# с чем работаем
model_path = Path(__file__).parent / "yolo26n-pose.pt"
sound_file = Path(__file__).parent / "zvuk.mp3"
model = YOLO(model_path)

out_path = Path(__file__).parent / "out"
out_path.mkdir(exist_ok=True)

cnt = 0
time_cnt = 0
prev_push = False

WAS_DOWN = False
DOWN_ANGLE_THRESHOLD = 100

camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(camera.get(cv2.CAP_PROP_FPS)) if camera.get(cv2.CAP_PROP_FPS) else 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
usb_output = out_path / 'usb_output.mp4'

video = cv2.VideoWriter(
    usb_output,
    fourcc,
    fps,
    (frame_width, frame_height)
)

while camera.isOpened():
    ret, frame = camera.read()
    cv2.imshow("camera", frame)

    key = cv2.waitKey(10) & 0xFF

    if key == ord("q"):
        break

    t = time.perf_counter()
    results = model(frame)
    print(f"FPS: {1 / (time.perf_counter() - t):.2f}")

    if not results:
        continue

    res = results[0]

    keypoints = res.keypoints.xy.tolist()
    if not keypoints:
        time_cnt += 1
        if time_cnt >= 100:
            time_cnt = 0
            prev_push=False
            cnt=0
    else:
        time_cnt = 0

    if keypoints:
        annotator = Annotator(frame)
        annotator.kpts(res.keypoints.data[0], res.orig_shape, 5, True)
        annotated = annotator.result()
    else:
        annotated = frame

    cv2.putText(annotated,
        f"cnt {cnt}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        2
    )

    if keypoints:
        cur_push = pushup(keypoints[0])

        if not cur_push:
            if get_avg_angle(keypoints) < DOWN_ANGLE_THRESHOLD:
                WAS_DOWN = True

        if cur_push and WAS_DOWN and not prev_push:
            playsound(sound_file, block=False)
            cnt += 1
            WAS_DOWN = False
        if cur_push is not None:
            prev_push = cur_push
    else:
        prev_push = False

    video.write(annotated)

    cv2.imshow("pose", annotated)

camera.release()
video.release()
cv2.destroyAllWindows()

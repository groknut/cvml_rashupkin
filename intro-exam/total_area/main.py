
import cv2
from pathlib import Path

images_path = Path(__file__).parent / "dataset"
images = [item for item in images_path.iterdir() if item.is_file()]

for image in images:
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sum_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 300)

    print("{}: {}".format(image.name, sum_area))

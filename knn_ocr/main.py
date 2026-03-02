

from pathlib import Path
import numpy as np
import matplotlib.pyplot as pp
from skimage.measure import regionprops, label
from skimage.io import imread
import cv2

def extractor(image):
    if image.ndim == 2:
        binary = image
    else:
        gray = np.mean(image, 2).astype("u1")
        binary = gray > 10

    labeled = label(binary)
    props = max(regionprops(labeled), key=lambda r: r.area)
    return np.array([*props.moments_hu, props.eccentricity], dtype='f4')

def make_train(path):
    train, responses = [], []
    ncls = 0
    class_names = dict()

    for cls in sorted(path.glob("*")):
        ncls += 1

        if 's' in cls.name.lower() and len(cls.name) == 2:
            class_names[ncls] = cls.name[1:]
        else: 
            class_names[ncls] = cls.name

        for item in cls.glob("*.png"):
            train.append(extractor(imread(item)))
            responses.append(ncls)

    train = np.array(train, dtype="f4")
    responses = np.array(responses, dtype="f4")
    return train, responses, class_names

out_path = Path(__file__).parent / "task"
train_path = out_path / "train"
none_char = "?"
knn = cv2.ml.KNearest.create()

train, responses, class_names = make_train(train_path)
print(rf"""
train: {train.shape}
responses: {responses.shape}
class_names: {len(class_names)} 
""")

knn.train(train, cv2.ml.ROW_SAMPLE, responses=responses)

for image_path in sorted(out_path.glob('*.png')):
    image = imread(image_path)
    gray = image.mean(2)
    binary = gray > 10
    lb = label(binary)
    props = regionprops(lb)

    sorted_props = sorted(props, key=lambda r: r.centroid[1])
    prev_max_col = None

    message = ''

    for prop in sorted_props:
        # если это точка от i, пропускаем
        if prop.area < 300: continue

        # проверяем наличие пробела
        # берем x координату верхнего левого угла и x координату правого нижнего угла 
        _, min_col, _, max_col = prop.bbox

        if prev_max_col is not None:
            if min_col - prev_max_col > 25:
                message += ' '
        
        char_feat = extractor(prop.image).reshape(1, -1)
        ret, results, neights, dist = knn.findNearest(char_feat, k=5)
        
        char = class_names.get(int(results[0][0]), none_char)
        message += char
        
        prev_max_col = max_col

    print(message)

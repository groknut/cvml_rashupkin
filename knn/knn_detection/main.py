

from pathlib import Path
import numpy as np
import matplotlib.pyplot as pp
from skimage.measure import regionprops, label
from skimage.io import imread
import cv2

# вернет параметры, которые передадим KNN
def extractor(image):
    if image.ndim == 2:
        binary = image
    else:
        gray = np.mean(image, 2).astype("u1")
        binary = gray < 255

    labeled = label(binary)
    props = regionprops(labeled)[0]

    # eccentricity - округлость
    # area - площадь
    return np.array([
        props.eccentricity, (props.area / np.pi) ** 0.5, 
    ], dtype='f4')

def make_train(path):
    train = []
    responses = []
    ncls = 0
    for cls in sorted(path.glob("*")):
        print(ncls)
        ncls += 1
        for item in cls.glob("*.png"):
            print(item)
            train.append(extractor(imread(item)))
            responses.append(ncls)
    train = np.array(train, dtype="f4").reshape(-1, 2)
    responses = np.array(responses, dtype="f4").reshape(-1, 1)
    return train, responses

out_path = Path(__file__).parent / "out"
train_path = out_path / "train"
knn = cv2.ml.KNearest.create()
train, responses = make_train(train_path)
print(r"""train: {}
responses: {}
""".format(train.shape, responses.shape))
knn.train(train, cv2.ml.ROW_SAMPLE, responses=responses)

test_image = imread(out_path / "image.png")

gray = test_image.mean(2)
binary = gray < 255
lb = label(binary)
find = []
props = regionprops(lb)

for prop in props:
    find.append(extractor(prop.image))

find = np.array(find, dtype="f4").reshape(-1, 2)

ret, results, neighs, dist = knn.findNearest(
    find,
    k=5
)

print(r"""ret: {}
results: {}
neighs: {}
dist: {}""".format(ret, results, len(neighs), dist))

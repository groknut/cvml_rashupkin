

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as pp

n = 100

x1 = 110 + np.random.randint(-25, 25, n)
y1 = 110 + np.random.randint(-25, 25, n)
r1 = np.repeat(1, n)

x2 = 150 + np.random.randint(-25, 25, n)
y2 = 150 + np.random.randint(-25, 25, n)
r2 = np.repeat(2, n)

new_point = (127, 124)
knn = cv2.ml.KNearest.create()
train = np.stack([np.hstack([x1, x2]), np.hstack([y1, y2])]).T.astype("f4")
responses = np.hstack([r1, r2]).reshape(-1, 1).astype("f4")
knn.train(train, cv2.ml.ROW_SAMPLE, responses=responses)

print(r"""train: {}
responses: {}
""".format(train.shape, responses.shape))

ret, results, neighs, dist = knn.findNearest(
    np.array(new_point).astype('f4').reshape(1, 2),
    k=3
)

print(r"""ret: {}
results: {}
neighs: {}
dist: {}""".format(ret, results, neighs, dist))

marker = "^"
if ret == 5:
    marker = "s"

pp.scatter(new_point[0], new_point[1], 80, "g", marker=marker)
pp.scatter(x1, y1, 80, "r", "^")
pp.scatter(x2, y2, 80, "b", "s")
pp.show()

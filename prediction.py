import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models

img_height = 28
img_width = 28
batch_size = 32

model = models.load_model("cracks.model")

class_name = ["not cracked", "cracked"]

img = cv.imread(f"img2.jpg")
img = cv.resize(img, (img_height, img_width))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img, cmap=plt.cm.binary)
img = np.array([img])
predict = model.predict(img, batch_size=batch_size)

# If is more than 0.5 it is a crack else it is not a crack
print(f"Stone is {class_name[int(np.round(predict))]}")

plt.show()

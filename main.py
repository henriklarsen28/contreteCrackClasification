import keras.utils
import tensorflow as tf
from tensorflow.python.keras import layers, models


# Declares image size and batch size
img_height = 28
img_width = 28
batch_size = 32

# Declares training images
training_images = keras.utils.image_dataset_from_directory(
    directory="Images/",
    labels="inferred",
    label_mode="binary",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Declares validation images
validation_images = keras.utils.image_dataset_from_directory(
    directory="Images/",
    labels="inferred",
    label_mode="binary",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

training_images = training_images
validation_images = validation_images

# Adds layers on model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))

model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(training_images, epochs=3, validation_data=validation_images)

model.save("cracks.model")


import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
import os

import tensorflow as tf
from keras import preprocessing
from keras.applications.efficientnet_v2 import preprocess_input, EfficientNetV2B1


if __name__ == "__main__":

    model = EfficientNetV2B1(weights="imagenet", include_top=False, pooling="avg")

    model.trainable = True  # or freeze some layers first

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_ds, epochs=...)

    model.save_weights("new_weights.h5")




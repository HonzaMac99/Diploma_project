"""
NIMA Training Script — VGG16 Backbone
======================================
Dataset structure expected:
    dataset/
        images/
            img1.jpg
            img2.jpg
            ...
        my_calc/aes_scores.csv   !! we have just one score, expected: (columns: image_id, score_1 ... score_10)

label CSV example:
    image_id,score_1,score_2,score_3,score_4,score_5,score_6,score_7,score_8,score_9,score_10
    img1.jpg,2,5,17,42,91,134,98,43,12,6
    ...

Each row contains the histogram of human ratings (1–10).
The script normalises them to a probability distribution internally.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30/"
RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

# todo: connect with dataset_loader.py

# ─────────────────────────────────────────────
# CONFIG — edit these paths / hyperparameters
# ─────────────────────────────────────────────
CONFIG = {
    "images_dir":    Path(DATASET_PATH),   # folder with all images
    "labels_csv":    Path(DATASET_PATH) / "my_calc/aes_scores.csv",
    "image_size":    (224, 224),         # VGG16 native input size
    "batch_size":    32,
    "epochs":        50,
    "learning_rate": 3e-4,
    "dropout":       0.75,
    "val_split":     0.1,                # fraction for validation
    "test_split":    0.1,                # fraction for test
    "weights_dir":   "weights",          # where checkpoints are saved
    "log_dir":       "logs",             # TensorBoard logs
    "freeze_base":   True,               # freeze VGG16 conv layers initially
    "unfreeze_epoch": 10,                # epoch at which to unfreeze & fine-tune
    "unfreeze_lr":   1e-5,              # lower LR after unfreezing
}

SCORE_BINS = np.arange(1, 11)           # scores 1 – 10


# ─────────────────────────────────────────────
# EARTH MOVER'S DISTANCE LOSS
# ─────────────────────────────────────────────
def earth_movers_distance(y_true, y_pred):
    """
    EMD loss: cumulative sum of the difference between CDF of true & pred.
    Better than cross-entropy for ordered score distributions.
    """
    cdf_true = tf.cumsum(y_true, axis=-1)
    cdf_pred = tf.cumsum(y_pred, axis=-1)
    return tf.reduce_mean(tf.reduce_mean(tf.square(cdf_true - cdf_pred), axis=-1))


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def mean_score(dist: np.ndarray) -> float:
    """Expected value of a score distribution."""
    return float(np.dot(dist, SCORE_BINS))


def std_score(dist: np.ndarray) -> float:
    """Standard deviation of a score distribution."""
    mean = mean_score(dist)
    return float(np.sqrt(np.dot(dist, (SCORE_BINS - mean) ** 2)))


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    score_cols = [f"score_{i}" for i in range(1, 11)]
    # Normalise histogram → probability distribution
    df[score_cols] = df[score_cols].div(df[score_cols].sum(axis=1), axis=0)
    return df


class NimaDataset(tf.keras.utils.Sequence):
    def __init__(self, df: pd.DataFrame, images_dir: str,
                 image_size: tuple, batch_size: int, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        self.score_cols = [f"score_{i}" for i in range(1, 11)]

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []

        for _, row in batch.iterrows():
            img_path = os.path.join(self.images_dir, row["image_id"])
            img = load_img(img_path, target_size=self.image_size)
            x = img_to_array(img)

            if self.augment:
                x = self._augment(x)

            # VGG16 preprocessing: zero-centre per ImageNet channel stats
            x = tf.keras.applications.vgg16.preprocess_input(x)
            images.append(x)
            labels.append(row[self.score_cols].values.astype(np.float32))

        return np.array(images), np.array(labels)

    @staticmethod
    def _augment(x: np.ndarray) -> np.ndarray:
        """Simple augmentation: random flip + slight brightness jitter."""
        if np.random.rand() > 0.5:
            x = np.fliplr(x)
        factor = np.random.uniform(0.85, 1.15)
        x = np.clip(x * factor, 0, 255)
        return x


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_nima_vgg16(dropout: float = 0.75, freeze_base: bool = True) -> Model:
    """
    VGG16 feature extractor + NIMA head.
    Output: 10-dim softmax (score probability distribution).
    """
    base = VGG16(
        weights="imagenet",
        include_top=False,          # drop VGG's original Dense head
        input_shape=(224, 224, 3),
    )
    base.trainable = not freeze_base

    x = Flatten()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(10, activation="softmax")(x)   # 10 score bins

    model = Model(inputs=base.input, outputs=x, name="NIMA_VGG16")
    return model, base


# ─────────────────────────────────────────────
# UNFREEZE CALLBACK
# ─────────────────────────────────────────────
class UnfreezeCallback(tf.keras.callbacks.Callback):
    """Unfreeze the base model at a specified epoch and lower the LR."""
    def __init__(self, base_model, unfreeze_epoch: int, new_lr: float):
        super().__init__()
        self.base_model = base_model
        self.unfreeze_epoch = unfreeze_epoch
        self.new_lr = new_lr
        self._unfrozen = False

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.unfreeze_epoch and not self._unfrozen:
            print(f"\n[UnfreezeCallback] Unfreezing VGG16 base at epoch {epoch}. LR → {self.new_lr}")
            self.base_model.trainable = True
            tf.keras.backend.set_value(self.model.optimizer.lr, self.new_lr)
            self._unfrozen = True


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train(cfg: dict):
    os.makedirs(cfg["weights_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)

    # ── Load & split labels ──────────────────
    df = load_labels(cfg["labels_csv"])
    train_val_df, test_df = train_test_split(
        df, test_size=cfg["test_split"], random_state=42
    )
    val_ratio = cfg["val_split"] / (1 - cfg["test_split"])
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, random_state=42
    )
    print(f"Split → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ── Datasets ─────────────────────────────
    train_ds = NimaDataset(train_df, cfg["images_dir"], cfg["image_size"],
                           cfg["batch_size"], augment=True)
    val_ds   = NimaDataset(val_df,   cfg["images_dir"], cfg["image_size"],
                           cfg["batch_size"], augment=False)
    test_ds  = NimaDataset(test_df,  cfg["images_dir"], cfg["image_size"],
                           cfg["batch_size"], augment=False)

    # ── Model ────────────────────────────────
    model, base_model = build_nima_vgg16(
        dropout=cfg["dropout"],
        freeze_base=cfg["freeze_base"],
    )
    model.compile(
        optimizer=Adam(learning_rate=cfg["learning_rate"]),
        loss=earth_movers_distance,
    )
    model.summary()

    # ── Callbacks ────────────────────────────
    checkpoint_path = os.path.join(cfg["weights_dir"], "nima_vgg16_best.keras")
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=cfg["log_dir"]),
    ]

    if cfg["freeze_base"]:
        callbacks.append(
            UnfreezeCallback(
                base_model=base_model,
                unfreeze_epoch=cfg["unfreeze_epoch"],
                new_lr=cfg["unfreeze_lr"],
            )
        )

    # ── Fit ──────────────────────────────────
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=callbacks,
    )

    # ── Evaluate on test set ─────────────────
    print("\n── Test set evaluation ──")
    test_loss = model.evaluate(test_ds, verbose=1)
    print(f"Test EMD loss: {test_loss:.6f}")

    # ── Sample predictions ───────────────────
    print("\n── Sample predictions (first batch) ──")
    x_sample, y_sample = test_ds[0]
    preds = model.predict(x_sample)
    for i in range(min(5, len(preds))):
        pred_mean = mean_score(preds[i])
        pred_std  = std_score(preds[i])
        true_mean = mean_score(y_sample[i])
        print(f"  [{i}] True mean: {true_mean:.2f} | Pred mean: {pred_mean:.2f} ± {pred_std:.2f}")

    # ── Save final weights ───────────────────
    final_path = os.path.join(cfg["weights_dir"], "nima_vgg16_final.keras")
    model.save(final_path)
    print(f"\nModel saved → {final_path}")
    return model, history


# ─────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────
def predict_image(model: Model, img_path: str, image_size: tuple = (224, 224)):
    """Score a single image and return mean ± std."""
    img = load_img(img_path, target_size=image_size)
    x = img_to_array(img)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    dist = model.predict(x, verbose=0)[0]
    return {
        "distribution": dist.tolist(),
        "mean":         mean_score(dist),
        "std":          std_score(dist),
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model, history = train(CONFIG)

    # Quick single-image demo (update path as needed)
    # result = predict_image(model, "dataset/images/sample.jpg")
    # print(f"Aesthetic score: {result['mean']:.2f} / 10 (±{result['std']:.2f})")
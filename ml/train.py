import warnings

warnings.filterwarnings(
    "ignore",
    message='Field "model_name" has conflict with protected namespace "model_".*',
)

import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf

# ---------- Config ----------
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT = os.getenv("EXPERIMENT_NAME", "fruit-freshness")

TRAIN_DIR = os.getenv("TRAIN_DIR", "data/train")
VAL_DIR = os.getenv("VAL_DIR", "data/val")

IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS = 5
LR = 1e-4

EXPORT_PATH = os.getenv(
    "EXPORT_MODEL_PATH", "models/latest_model.keras"
)  # for FastAPI use later


def build_model():
    # Simple + fast baseline (good for first-time teaching)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # first run: freeze backbone

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(
        x
    )  # 0=fresh, 1=rotten (we will treat >0.5 as rotten)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
        shuffle=False,
    )

    # Performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run():
        # Log params (super important for teaching)
        mlflow.log_param("img_size", str(IMG_SIZE))
        mlflow.log_param("batch_size", BATCH)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("lr", LR)
        mlflow.log_param("train_dir", TRAIN_DIR)
        mlflow.log_param("val_dir", VAL_DIR)
        mlflow.log_param("model", "MobileNetV2_frozen")

        train_ds, val_ds = load_data()
        model = build_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )

        # Log final metrics
        mlflow.log_metric("train_acc_last", float(history.history["accuracy"][-1]))
        mlflow.log_metric("val_acc_last", float(history.history["val_accuracy"][-1]))
        mlflow.log_metric("train_auc_last", float(history.history["auc"][-1]))
        mlflow.log_metric("val_auc_last", float(history.history["val_auc"][-1]))

        # Evaluate & log
        val_loss, val_acc, val_auc = model.evaluate(val_ds, verbose=0)
        mlflow.log_metric("val_loss", float(val_loss))
        mlflow.log_metric("val_acc", float(val_acc))
        mlflow.log_metric("val_auc", float(val_auc))

        # Log model to MLflow (artifact)
        mlflow.tensorflow.log_model(model, artifact_path="model")

        # Export a copy into /models for FastAPI to load easily
        os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
        model.save(EXPORT_PATH)
        mlflow.log_artifact(EXPORT_PATH)

        print(f"\n✅ Exported model for backend: {EXPORT_PATH}")


if __name__ == "__main__":
    main()

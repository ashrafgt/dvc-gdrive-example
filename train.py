import tensorflow as tf
from transform import parse_tfrecord
import sys


def prepare_sample(features):
    image = tf.image.resize(features["image"], size=(224, 224))
    return image, features["category_id"]


def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset


if __name__ == "__main__":

    model_name = sys.argv[1]
    models_dir_path = "models/checkpoints/" + model_name + ".h5"
    tfrecords_dir_path = "data/tfrecords"

    batch_size = 32
    epochs = 1
    steps_per_epoch = 50
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_filenames = tf.io.gfile.glob(f"{tfrecords_dir_path}/*.tfrec")

    input_tensor = tf.keras.layers.Input(shape=(224, 224, 3), name="image")
    model = tf.keras.applications.EfficientNetB0(
        input_tensor=input_tensor, weights=None, classes=91
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        x=get_dataset(train_filenames, batch_size),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
    )

    model.save(models_dir_path, save_format="h5")

import json
import math
import tensorflow as tf


def create_example(image, path, example):
    feature = {
        "image": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.encode_jpeg(image).numpy()]
            )
        ),
        "path": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[path.encode()])
        ),
        "area": tf.train.Feature(
            float_list=tf.train.FloatList(value=[example["area"]])
        ),
        "bbox": tf.train.Feature(
            float_list=tf.train.FloatList(value=example["bbox"])
        ),
        "category_id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[example["category_id"]])
        ),
        "id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[example["image_id"]])
        ),
        "image_id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[example["image_id"]])
        ),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "area": tf.io.FixedLenFeature([], tf.float32),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    return example


if __name__ == "__main__":

    annotations_file_path = "data/raw/annotations/instances_val2017.json"
    max_records = 5000
    tfrecords_dir_path = "data/tfrecords"
    images_dir_path = "data/raw/images"

    with open(annotations_file_path, "r") as f:
        annotations = json.load(f)["annotations"][0:max_records]

    num_samples = 4096
    num_tfrecods = math.ceil(len(annotations) / num_samples)

    for tfrec_num in range(num_tfrecods):
        samples = annotations[
            (tfrec_num * num_samples): ((tfrec_num + 1) * num_samples)
        ]

        with tf.io.TFRecordWriter(
            tfrecords_dir_path + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
        ) as writer:
            for sample in samples:
                image_path = f"{images_dir_path}/{sample['image_id']:012d}.jpg"
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                example = create_example(image, image_path, sample)
                writer.write(example.SerializeToString())

    raw_dataset = tf.data.TFRecordDataset(
        f"{tfrecords_dir_path}/file_00-{num_samples}.tfrec"
    )
    parsed_dataset = raw_dataset.map(parse_tfrecord)

    for features in parsed_dataset.take(1):
        for key in features.keys():
            if key != "image":
                print(f"{key}: {features[key]}")
        print(f"Image shape: {features['image'].shape}")
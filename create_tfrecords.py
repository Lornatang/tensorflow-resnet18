import os
from PIL import Image
import tensorflow as tf


# write mnist to tfrecords.
def create_tfrecords(path, name):
    cwd = os.getcwd()
    classes = os.listdir(cwd + path)

    writer = tf.python_io.TFRecordWriter(name)
    for index, name in enumerate(classes):
        img_dir = cwd + path + name + "/"
        if os.path.isdir(img_dir):
            for img_name in os.listdir(img_dir):
                img = Image.open(img_dir + img_name)
                img = img.resize((32, 32))
                img = img.tobytes()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[int(name)])),
                            'data': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[img]))
                        }))
                writer.write(example.SerializeToString())
    writer.close()


create_tfrecords(path="/train/", name="train.tfrecords")
create_tfrecords(path="/val/", name="val.tfrecords")

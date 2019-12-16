import numpy as np
import scipy as sc
import scipy.signal as sig
import time
import random
import string
import os
import math
import os
from pathlib import Path
from progress.bar import Bar
import tensorflow as tf
import hashlib
import pandas as pd

from sklearn.model_selection import train_test_split

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("input_path", "rfcnn-data", "Path to data")
flags.DEFINE_string("output_path", "rfcnn-training", "Path to data")
flags.DEFINE_integer(
    "input_dim", "28", "Length of side to square input dimension")
flags.DEFINE_integer("downsamp_factor", "10",
                     "Length of side to square input dimension")


def __randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def ContainsSignal(data, threshold=600, min_idx=200, max_idx=584*2):
    'Simple numeric search to see if a signal goes over a certain value'
    locs = np.where(data.flatten() > threshold)[0]
    if not locs.size:
        return False
    if locs[0] < max_idx and locs[len(locs)-1] > min_idx:
        return True
    return False


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def ConvertFile(file_path, label, label_idx, output_dir):
    iq_samples = np.fromfile(str(file_path), dtype=np.int16)
    length = math.floor(len(iq_samples)/(FLAGS.input_dim*FLAGS.input_dim*2))
    iq_samples.resize((length, FLAGS.input_dim, FLAGS.input_dim, 2), refcheck=False)
    
    # Filter for signal present in each frame
    iq_samples = iq_samples[np.array([ContainsSignal(row) for row in iq_samples])]
    
    output_dir = os.path.join(output_dir, label)
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    base=os.path.basename(file_path)
    output_file = os.path.join(output_dir, os.path.splitext(base)[0] + ".tfrecord")

    writer = tf.io.TFRecordWriter(output_file)

    x = tf.convert_to_tensor(iq_samples)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[FLAGS.input_dim])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[FLAGS.input_dim])),
        'image/num': tf.train.Feature(int64_list=tf.train.Int64List(value=[iq_samples.shape[0]])),
        'image/encoded': _bytes_feature(tf.io.serialize_tensor(x)),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['int16'.encode('utf8')])),
        'image/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_idx]))
        }))
    writer.write(example.SerializeToString())


    writer.close()  

    # Convert to np.complex64 for decimation
    # iq_samples = iq_samples.astype(np.float32).view(np.complex64)

    # De-interleave real and imaginary data then decimate them seperately
    # iq_dec = sig.decimate(iq_samples, 10, ftype='fir')
    # iq_real = sig.decimate(
    #     iq_samples[0::2], FLAGS.downsamp_factor, ftype='fir')
    # iq_complex = sig.decimate(
    #     iq_samples[1::2], FLAGS.downsamp_factor, ftype='fir')
    # del iq_samples
    # reinterleave data into 2D numpy array
    # iq_out = np.column_stack((iq_real, iq_complex))


def main(argv):
    data_files = [(str(f), f.parts[-2].split('-')[0]) for f in Path(FLAGS.input_path).rglob('*.xdat') if f]
    df = pd.DataFrame(data_files, columns=['path', 'label'])
    bar = Bar('Processing', max=len(df))

    classes = df.label.unique()

    train_data, val_data = train_test_split(df, train_size=.35, test_size=.15, stratify=df['label'])        

    output_dir = os.path.join(FLAGS.output_path, 'train')
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    output_dir = os.path.join(FLAGS.output_path, 'val')
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    for _, row in train_data.iterrows():
        class_idx = np.where(classes == row.label)[0][0]
        ConvertFile(row.path, row.label, class_idx, os.path.join(FLAGS.output_path, 'train'))
        bar.next()
    for _, row in val_data.iterrows():
        class_idx = np.where(classes == row.label)[0][0]
        ConvertFile(row.path, row.label, class_idx, os.path.join(FLAGS.output_path, 'val'))
        bar.next()
    bar.finish()

    with open(os.path.join(FLAGS.output_path, 'your_file.txt'), 'w') as f:
        for item in classes:
            f.write("%s\n" % item)
    bar.finish()
              

if __name__ == '__main__':
  app.run(main)



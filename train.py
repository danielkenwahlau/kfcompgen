#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import math
import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import time
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Input, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


# In[2]:


# Define console flags
trainPath = "rfcnn-training"
batchSize = 128

totalsamples = 0
# target values are labels persample

target_samples = []
#batch = 0


# In[3]:


def get_data(path):
    # ''' Returns dataframe with columns: 'path', 'word'.'''
    datadir = Path(path)
    files = [(str(f), f.parts[-2].split('-')[0]) for f in datadir.glob('**/*.tfrecord') if f]
    df = pd.DataFrame(files, columns=['path', 'word'])

    return df


# In[4]:


IMAGE_FEATURE_MAP = {
    'image/width': tf.io.FixedLenFeature([1], tf.int64),
    'image/height': tf.io.FixedLenFeature([1], tf.int64),
    'image/num': tf.io.FixedLenFeature([1], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/label': tf.io.FixedLenFeature([1], tf.int64),
}

def LoadTFRecord(tfrecord):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    # Might need to shard numpy file to save memory or run eagerly
    data = tf.io.parse_tensor(x['image/encoded'], out_type=tf.int16)
    shape = tf.concat([
        x['image/num'],
        x['image/width'],
        x['image/height'],
        tf.constant([2], dtype=tf.int64)
        ], axis=0)
    data = tf.reshape(data, shape)
#     data = tf.dtypes.cast(data, dtype=tf.float32)
    y_train = tf.fill(x['image/num'], tf.squeeze(x['image/label']))
    return tf.data.Dataset.from_tensor_slices((data, y_train))

def CreateDataset(dataframe):
    num_files = len(dataframe.index)
    files = tf.data.Dataset.from_tensor_slices(dataframe['path'].values)
    files = files.shuffle(num_files)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.interleave(lambda tfrecord: LoadTFRecord(tfrecord), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batchSize)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.repeat()

    return dataset


# In[5]:


# pass the number of classes or directory labels
def get_model(word_length):
    dim1 = 28
    dim2 = 28
    # in this case we are essenitally passing a 28 by 28 array of IQ data
    # next we make a 3X3 window that slides over the array looking for "features"
    input_signal = Input(shape=(dim1, dim2, 2))
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_signal)
    # in general you want to use drop out to decrease overfitting
    # experimentation with adding drop out to layers or all layers and at
    # what setting is important although there are a lot of established best practices
    # x = Dropout(0.2)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(word_length, activation='softmax')(x)

    model = Model(inputs=input_signal, outputs=x)
    model.summary()
    return model


# In[8]:


train_data = get_data(os.path.join(trainPath, 'train'))
val_data = get_data(os.path.join(trainPath, 'val'))

# Encode class label to integer 
le = LabelEncoder()
train_data["class_int"] = le.fit_transform(train_data.word)
val_data["class_int"] = le.fit_transform(val_data.word)

num_classes = len(train_data.word.unique())

train_ds = CreateDataset(train_data)
val_ds = CreateDataset(val_data)


# In[ ]:


# Create tensorboard outputs
log_dir=os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

model = get_model(num_classes)

metrics = [
    'accuracy',
    'sparse_categorical_accuracy'
]
callbacks = [
    ReduceLROnPlateau(verbose=1, patience=1),
#     EarlyStopping(patience=2, verbose=1),
#     ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
#                     verbose=1, save_weights_only=True),
    TensorBoard(log_dir=log_dir)
]

opt = tf.keras.optimizers.Adam()
# opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
model.compile(loss='sparse_categorical_crossentropy',
            optimizer=opt, metrics=metrics)

model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[callbacks],
#     validation_steps=3000,
#     steps_per_epoch=10000,
)


# In[ ]:


acc = tf.keras.metrics.Accuracy()
cat_acc = tf.keras.metrics.CategoricalAccuracy()
mean_err = tf.keras.metrics.MeanAbsoluteError()

predictions = np.array([])
truth = np.array([])
for x in val_ds.take(1000):
    pred = model.predict(tf.cast(x[0], dtype=tf.float32), batch_size=batchSize, verbose=0)
    predictions = np.append(predictions, np.argmax(pred, axis=1))
    truth = np.append(truth, x[1].numpy())

    acc.update_state(x[1].numpy(), np.argmax(pred, axis=1))
    cat_acc.update_state(tf.one_hot(tf.cast(x[1], dtype=tf.int32), num_classes), pred)
    mean_err.update_state(x[1].numpy(), pred[:,np.argmax(pred, axis=1)])


print('Accuracy: ', acc.result().numpy())
print('Categorical Accuracy: ', cat_acc.result().numpy())
print('mean_err: ', mean_err.result().numpy())

y_target_names = le.inverse_transform(range(num_classes))

print(classification_report(truth, predictions, target_names=y_target_names))
cm = confusion_matrix(truth, predictions)
print(cm)

df_cm = pd.DataFrame(cm, index=y_target_names,
            columns=y_target_names)
sn.set(font_scale=1)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12}, fmt='d', cbar=False)# font size
plt.ylabel('predicted')
plt.xlabel('target')
plt.show()


# In[ ]:





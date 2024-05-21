"""
Converts google competition landmarks into 
"""

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import os

from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import cpu_count
from sklearn.model_selection import StratifiedGroupKFold, KFold

import tensorflow as tf
from const import *

print(cpu_count())

ROOT = os.path.expanduser("~")
MAIN_PATH = ROOT + "/Documents/Tesis/Datasets/GoogleCompetitition"
DATA_PATH = MAIN_PATH + '/train_landmark_files'
LABELS = "train.csv"
ROWS_PER_FRAME = 543
DATASET_NAME = "GoogleASLDataset"

labels_path = os.path.join(MAIN_PATH, LABELS)
train_df = pd.read_csv(labels_path)
frames = []


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    # print(f'file: {pq_path} | frames: {n_frames}')
    frames.append(n_frames)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))

    return data.astype(np.float32)


with open(os.path.join(MAIN_PATH, 'sign_to_prediction_index_map.json')) as json_file:
    LABEL_DICT = json.load(json_file)


def encode_row(row):
    coordinates = load_relevant_data_subset(os.path.join(MAIN_PATH, row.path))
    coordinates_encoded = coordinates.tobytes()
    participant_id = int(row.participant_id)
    sequence_id = int(row.sequence_id)
    sign = int(LABEL_DICT[row.sign])

    record_bytes = tf.train.Example(features=tf.train.Features(feature={
        'coordinates': tf.train.Feature(bytes_list=tf.train.BytesList(value=[coordinates_encoded])),
        'participant_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[participant_id])),
        'sequence_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_id])),
        'sign': tf.train.Feature(int64_list=tf.train.Int64List(value=[sign])),
    })).SerializeToString()

    return record_bytes





# Test first parquet file (landmarks for a sign) of csv row
row = train_df.iloc[0]
coordinates = load_relevant_data_subset(os.path.join(MAIN_PATH, row.path))
coordinates_encoded = coordinates.tobytes()
participant_id = int(row.participant_id)
sequence_id = int(row.sequence_id)
sign = int(LABEL_DICT[row.sign])

# ---

N_FILES = len(train_df)
CHUNK_SIZE = 512
N_PART = 1
FOLD = 4
part = 0


class CFG:
    seed = 42
    n_splits = 4


train_folds = train_df.copy()
# Se le crea una columna fold al csv
train_folds['fold'] = -1

num_bins = 5


# Put every image in a seperate TFRecord file
# Make Pairs of Views as input to the model
def split_dataframe(df, chunk_size=10000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks


def process_chunk(chunk, tfrecord_name):
    options = tf.io.TFRecordOptions(compression_type='GZIP', compression_level=9)

    with tf.io.TFRecordWriter(tfrecord_name, options=options) as file_writer:
        for i, row in tqdm(chunk.iterrows()):
            record_bytes = encode_row(row)
            file_writer.write(record_bytes)
            del record_bytes
        file_writer.close()


# Splits dataset into n_split consecutive folds and provides train/test indices to split data in train/tests sets.
kfold = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
print(f'{CFG.n_splits}fold training', len(train_folds), 'samples')

for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(train_folds)):
    train_folds.loc[valid_idx, 'fold'] = fold_idx
    print(f'fold{fold_idx}:', 'train', len(train_idx), 'valid', len(valid_idx))

assert not (train_folds['fold'] == -1).sum()
assert len(np.unique(train_folds['fold'])) == CFG.n_splits
print("train folds head ", train_folds.head())


for fold in range(CFG.n_splits):
    # selects rows from the train_folds csv where the 'fold' column value matches the current fold (1, 2, 3, or 4)
    rows = train_folds[train_folds['fold'] == fold]
    chunks = split_dataframe(rows, CHUNK_SIZE)
    part_size = len(chunks) // N_PART
    last = (part + 1) * part_size if part != N_PART - 1 else len(chunks) + 1
    chunks = chunks[part * part_size:last]

    N = [len(x) for x in chunks]

    _ = Parallel(n_jobs=cpu_count())(
        delayed(process_chunk)(x, f'/tmp/{DATASET_NAME}/fold{fold}-{i}-{n}.tfrecords')
        for i, (x, n) in enumerate(zip(chunks, N))
    )

x_axis = range(len(frames))
plt.bar(x_axis, frames)

plt.xlabel("Sign")
plt.ylabel("Amount of frames")
plt.title("Bar Chart of frames per sign")

# Show the plot
plt.show()

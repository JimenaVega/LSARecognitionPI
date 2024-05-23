"""
Converts google competition landmarks into 
"""

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import os

from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import cpu_count
from sklearn.model_selection import KFold

import tensorflow as tf

print(cpu_count())

ROOT = os.path.expanduser("~")
DATAPATH = ROOT + "/Documents/Tesis/Datasets/LSA64"
PARQUETS_PATH = DATAPATH + '/parquets'
ROWS_PER_FRAME = 543

CHUNK_SIZE = 512
N_PART = 1
FOLD = 4
part = 0
frames = []


class CFG:
    seed = 42
    n_splits = 4


def load_relevant_data_subset(pq_path):
    """
    Se lee el .parquet file indificado por la row del csv y se cargan las
    columnas de los landmarks.
    Luego se hace un reshape de los datos (frames x 543 landmarks x 3)
    """
    data_columns = ['x', 'y', 'z']
    try:
        data = pd.read_parquet(pq_path, columns=data_columns)
    except:
        raise Exception(f"File {pq_path} couldnt be read")

    n_frames = int(len(data) / ROWS_PER_FRAME)
    frames.append(n_frames)

    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))

    return data.astype(np.float32)


def encode_row(row):
    """
    Se codifican los datos de la row a formato tfRecords
    """
    row_path = os.path.join(DATAPATH, row.path)
    coordinates = load_relevant_data_subset(row_path)
    coordinates_encoded = coordinates.tobytes()
    participant_id = int(row.participant_id)
    sequence_id = int(row.sequence_id)
    print(f"ROW-> {row.sign}")
    # if math.isnan(row.sign):
    #     sign = 38
    # else:
    sign = int(LABEL_DICT[row.sign])

    record_bytes = tf.train.Example(features=tf.train.Features(feature={
        'coordinates': tf.train.Feature(bytes_list=tf.train.BytesList(value=[coordinates_encoded])),
        'participant_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[participant_id])),
        'sequence_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_id])),
        'sign': tf.train.Feature(int64_list=tf.train.Int64List(value=[sign])),
    })).SerializeToString()

    return record_bytes


# Put every image in a separate TFRecord file
# Make Pairs of Views as input to the model
def split_dataframe(df, chunk_size=10000):
    """
    Se dividen los datos del foldX en chunks.
    Por ejemplo si para fold0 tengo 23620 archivos, los divido en chunks de 512 y al final me quedan 47 chunks (47 tfrecords)
    """
    chunks = list()
    num_chunks = len(df) // chunk_size + 1

    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks


def process_chunk(chunk, tfrecord_name):
    print("tfrecord_name: ", tfrecord_name)
    options = tf.io.TFRecordOptions(compression_type='GZIP', compression_level=9)

    with tf.io.TFRecordWriter(tfrecord_name, options=options) as file_writer:
        for i, row in tqdm(chunk.iterrows()):
            print(f"ROW process chunk {row}")
            record_bytes = encode_row(row)
            file_writer.write(record_bytes)
            del record_bytes
        file_writer.close()


# Csv with all .parquet reading
LABELS_PATH = os.path.join(DATAPATH, 'parquets_data.csv')
train_df = pd.read_csv(LABELS_PATH)
print(train_df.head())
print(train_df.info())
N_FILES = len(train_df)

# Read first parquet
pd.read_parquet(os.path.join(PARQUETS_PATH, '1001001001.parquet'))

# Read labels file
JSON_LABELS = os.path.join(DATAPATH, 'labels.json')
with open(JSON_LABELS) as json_file:
    LABEL_DICT = json.load(json_file)


train_folds = train_df.copy()
train_folds['fold'] = -1  # Se le crea una columna fold al csv

num_bins = 5

# Splits dataset into n_split consecutive folds and provides train/test indices to split data in train/tests sets.
kfold = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
print(f'{CFG.n_splits}fold training', len(train_folds), 'samples')

# Se toman los indices de row indicados por valid_idx (list) y en la columna folds se le asigna el numero de fold_idx
for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(train_folds)):
    train_folds.loc[valid_idx, 'fold'] = fold_idx
    print(f'fold{fold_idx}:', 'train', len(train_idx), 'valid', len(valid_idx))

assert not (train_folds['fold'] == -1).sum()
assert len(np.unique(train_folds['fold'])) == CFG.n_splits

print(train_folds.head())

for fold in range(CFG.n_splits):
    # selects rows from the train_folds csv where the 'fold' column value matches the current fold (1, 2, 3, or 4)
    rows = train_folds[train_folds['fold'] == fold]
    chunks = split_dataframe(rows, CHUNK_SIZE)
    part_size = len(chunks) // N_PART
    last = (part + 1) * part_size if part != N_PART - 1 else len(chunks) + 1
    chunks = chunks[part * part_size:last]  # rows agrupadas en chunks

    N = [len(x) for x in chunks]

    _ = Parallel(n_jobs=cpu_count() - 8)(
        delayed(process_chunk)(x, f'{DATAPATH}/tfrecords/fold{fold}-{i}-{n}.tfrecords')
        # f'/tmp/{DATASET_NAME}/fold{fold}-{i}-{n}.tfrecords'
        for i, (x, n) in enumerate(zip(chunks, N))
    )

import os
import numpy as np
import random
import sys
import re
import tensorflow as tf

from const import DATAPATH


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename.split('/')[-1]).group(1)) for filename in filenames]
    return np.sum(n)


def get_strategy(cfg, device='TPU-VM'):
    IS_TPU = False
    if "TPU" in device:
        tpu = 'local' if device=='TPU-VM' else None
        print("connecting to TPU...")
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        IS_TPU = True

    if device == "GPU" or device=="CPU":
        ngpu = len(tf.config.experimental.list_physical_devices('GPU'))
        if ngpu>1:
            print("Using multi GPU")
            strategy = tf.distribute.MirroredStrategy()
        elif ngpu==1:
            print("Using single GPU")
            strategy = tf.distribute.get_strategy()
        else:
            print("Using CPU")
            strategy = tf.distribute.get_strategy()
            cfg.device = "CPU"

    if device == "GPU":
        print("Num GPUs Available: ", ngpu)

    AUTO     = tf.data.experimental.AUTOTUNE
    REPLICAS = strategy.num_replicas_in_sync
    print(f'REPLICAS: {REPLICAS}')

    return strategy, REPLICAS, IS_TPU


def check_weights_folder():
    if not os.path.isdir(DATAPATH + 'weights'):
        os.makedirs(DATAPATH + 'weights')


def check_gpu():
    gpu_info = ''
    gpu_info = '\n'.join(gpu_info)

    if gpu_info.find('failed') >= 0:
        print('Not connected to a GPU')
    else:
        print("Connected to GPU")
        print(gpu_info)

    print(f'Tensorflow Version: {tf.__version__}')
    print(f'Python Version: {sys.version}')

    device_name = tf.test.gpu_device_name()
    print(f'device name: {device_name}')

    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')

    print('Found GPU at: {}'.format(device_name))


# Seed all random number generators
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

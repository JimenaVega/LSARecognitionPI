# import tensorflow_addons as tfa # ver para   opt = tfa.optimizers.AdamW(learning_rate=schedule, weight_decay=decay_schedule)
# import keras_nlp

# from tensorflow.keras import mixed_precision # import tensorflow.keras.mixed_precision as mixed_precision

# from tf_utils.schedules import OneCycleLR, ListedLR
# from tf_utils.callbacks import Snapshot, SWA
# from tf_utils.learners import FGM, AWP
import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_addons as tfa
#import matplotlib.pyplot as plt
#import matplotlib as mpl
# import tensorflow.keras.mixed_precision as mixed_precision

from tqdm.autonotebook import tqdm
import sklearn

from tf_utils.schedules import OneCycleLR, ListedLR
from tf_utils.callbacks import Snapshot, SWA
from tf_utils.learners import FGM, AWP

import os
import time
import pickle
import math
import random
import sys
import cv2
import gc
import glob
import datetime
import re

ROWS_PER_FRAME = 543
MAX_LEN = 384
CROP_LEN = MAX_LEN
NUM_CLASSES = 250  # amount of labels
PAD = -100.

NOSE = [
    1, 2, 98, 327
]
LNOSE = [98]
RNOSE = [327]
LIP = [0,
       61, 185, 40, 39, 37, 267, 269, 270, 409,
       291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
       95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
       ]
LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
RLIP = [314, 405, 321, 375, 291, 409, 270, 269, 267, 317, 402, 318, 324, 308, 415, 310, 311, 312]

POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513, 505, 503, 501]
RPOSE = [512, 504, 502, 500]

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]

LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE  # +POSE

NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6 * NUM_NODES

print('nodos (landmarks len): ', NUM_NODES)
print('channels (landmarks len * 6) ', CHANNELS)

ROOT = os.path.expanduser("~")
DATAPATH = ROOT + "/Documents/Tesis/Datasets/GoogleCompetitition/"
TRAIN_FILENAMES = glob.glob(DATAPATH + '*.tfrecords')
print(TRAIN_FILENAMES)
print(len(TRAIN_FILENAMES))

# Read labels and file datapath
train_df = pd.read_csv(DATAPATH + 'train.csv')
print(train_df.head())
# display(train_df.tail())
print(train_df.info())

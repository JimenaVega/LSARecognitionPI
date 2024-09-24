import os
import csv
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from train import train_folds

from utils import get_strategy
from utils import count_data_items
from utils import get_strategy

from const import DATAPATH
from const import TRAININGPATH
from const import WEIGHTSPATH
from const import MAX_LEN
from const import TRAIN_FILENAMES


class CFG:
    files_index = None

    n_splits = 4
    save_output = True
    output_dir = WEIGHTSPATH

    seed = 42
    verbose = 2  # 0-silent 1-progress bar 2-one line per epoch

    max_len = MAX_LEN
    replicas = 8
    lr = 0.01#5e-4 * replicas # 0.01 para empezar
    weight_decay = 0.1
    lr_min = 1e-6
    epoch = 200
    warmup = 0
    batch_size = 128#64 * replicas
    snapshot_epochs = []
    swa_epochs = [] #list(range(epoch//2,epoch+1))

    fp16 = True
    fgm = False
    awp = True
    awp_lambda = 0.2
    awp_start_epoch = 15
    dropout_start_epoch = 15
    resume = 0
    decay_type = 'cosine'
    dim = 192
    comment = None

    transfer_learning = False
    load_weights = False
    export_model = False


if __name__ == "__main__":

    if not os.path.exists(TRAININGPATH + 'info.csv'):
        column_headers = ['best', 'last', 'logs', 'n_splits', 'seed', 'max_len', 'replicas', 'lr', 'epoch', 'batch_size', 'dim', 'transfer_learning', 'load_weights']

        with open(TRAININGPATH + 'info.csv', 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(column_headers)

    csv_file = csv.reader(open(TRAININGPATH + 'info.csv'))
    CFG.files_index = sum(1 for row in csv_file) - 1
    CFG.comment = f'lsa-{CFG.files_index}'

    # Train DataFrame
    train_df = pd.read_csv(DATAPATH + 'parquets_data_cut_nan.csv')

    print(f'Parquets count: {count_data_items(TRAIN_FILENAMES)} - Parquets CSV rows: {len(train_df)}')

    STRATEGY, N_REPLICAS, IS_TPU = get_strategy(CFG, 'GPU')
    #STRATEGY=None
    train_folds(CFG, [0], STRATEGY)


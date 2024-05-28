import pandas as pd

from train import train_folds

from utils import get_strategy
from utils import count_data_items
from utils import get_strategy

from const import DATAPATH
from const import MAX_LEN
from const import TRAIN_FILENAMES


class CFG:
    n_splits = 4
    save_output = True
    output_dir = DATAPATH + 'weights'

    seed = 42
    verbose = 2 # 0-silent 1-progress bar 2-one line per epoch

    max_len = MAX_LEN
    replicas = 8
    lr = 5e-4 * replicas
    weight_decay = 0.1
    lr_min = 1e-6
    epoch = 300 # 300 400
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
    comment = f'islr-fp16-192-8-seed{seed}'


if __name__ == "__main__":
    # Train DataFrame
    train_df = pd.read_csv(DATAPATH + 'parquets_data.csv')

    print(f'Parquets count: {count_data_items(TRAIN_FILENAMES)} - Parquets CSV rows: {len(train_df)}')

    STRATEGY, N_REPLICAS, IS_TPU = get_strategy(CFG, 'GPU')

    train_folds(CFG, [0], STRATEGY)


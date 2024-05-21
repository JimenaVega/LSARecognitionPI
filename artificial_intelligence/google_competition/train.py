from const import *


def train_fold(CFG, fold, train_files, valid_files=None, summary=True):
    # seed_everything(CFG.seed)
    pass


def train_folds(CFG, folds, summary=True):
    print("function train_foldsd")
    for fold in folds:
        if fold != 'all':
            all_files = TRAIN_FILENAMES
            train_files = [x for x in all_files if f'fold{fold}' not in x]
            for x in all_files:
                if f'fold{fold}' not in x:
                    print("hi")
            valid_files = [x for x in all_files if f'fold{fold}' in x]
        else:
            train_files = TRAIN_FILENAMES
            valid_files = None
        print("----------------")
        print(f'train_files = {train_files}')
        print(f'valid_files = {valid_files}')

        print("----------------")

        train_fold(CFG, fold, train_files, valid_files, summary=summary)
    return


class CFG:
    n_splits = 5
    save_output = True
    output_dir = DATAPATH + '/aslfr'  # '/kaggle/working'

    seed = 42
    verbose = 2  # 0-silent 1-progress bar 2-one line per epoch

    max_len = 384
    replicas = 8
    lr = 5e-4 * replicas
    weight_decay = 0.1
    lr_min = 1e-6
    epoch = 300  # 400
    warmup = 0
    batch_size = 64 * replicas
    snapshot_epochs = []
    swa_epochs = []  # list(range(epoch//2,epoch+1))

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


train_folds(CFG, [0])

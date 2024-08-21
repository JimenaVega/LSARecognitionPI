import os
import gc
import csv
import json
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.mixed_precision as mixed_precision  # type: ignore

from tf_utils.callbacks import Snapshot, SWA
from tf_utils.learners import FGM, AWP

from utils import seed_everything
from utils import count_data_items

from dotenv import load_dotenv
load_dotenv()

from data_process import get_tfrec_dataset

from model import get_model

from const import TRAIN_FILENAMES
from const import TRAININGPATH
from const import WEIGHTSPATH
from const import NUM_CLASSES

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback


run = neptune.init_run(
    project="alejosm/LSA-Interpreter",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNjQ5MzNkNS0yYjZiLTRjMzgtYWFkZS0xYmFlOWMwNzE1MjYifQ==",
)

neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')


CMPATH = os.getenv('CMPATH')

def log_confusion_matrix(model, valid_ds, num_classes, name):
    y_true = []
    y_pred = []

    for x, y in valid_ds:
        preds = model.predict(x)
        y_true.extend(np.argmax(y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    plt.figure(figsize=(38.4, 21.6), dpi=100)  # Ajusta el tamaño según sea necesario
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True, linecolor='black', linewidths=0.1)
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    # Guardar la figura
    plt.savefig(f"{CMPATH}{name}.png")

    # Subir la matriz a Neptune
    run["confusion_matrix"].upload(f"{CMPATH}{name}.png")


def train_fold(CFG, fold, train_files, strategy, valid_files=None, summary=True):
    seed_everything(CFG.seed)
    tf.keras.backend.clear_session()
    gc.collect()
    tf.config.optimizer.set_jit(True)

    if CFG.fp16:
        try:
            policy = mixed_precision.Policy('mixed_bfloat16')
            mixed_precision.set_global_policy(policy)
        except:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
    else:
        policy = mixed_precision.Policy('float32')
        mixed_precision.set_global_policy(policy)

    # Pre-procesamiento de datos
    if fold != 'all':
        train_ds = get_tfrec_dataset(train_files, batch_size=CFG.batch_size, max_len=CFG.max_len, drop_remainder=True,
                                     augment=True, repeat=True, shuffle=1000)
        valid_ds = get_tfrec_dataset(valid_files, batch_size=CFG.batch_size, max_len=CFG.max_len, drop_remainder=False,
                                     repeat=False, shuffle=False)
    else:
        train_ds = get_tfrec_dataset(train_files, batch_size=CFG.batch_size, max_len=CFG.max_len, drop_remainder=False,
                                     augment=True, repeat=True, shuffle=1000)  # augment=True
        valid_ds = None
        valid_files = []

    num_train = count_data_items(train_files)
    num_valid = count_data_items(valid_files)
    steps_per_epoch = num_train // CFG.batch_size

    print("----------------")
    print("num train: ", num_train)
    print("num valid: ", num_valid)
    print("steps_per_epoch", steps_per_epoch)
    print("----------------")

    with strategy.scope():
        dropout_step = CFG.dropout_start_epoch * steps_per_epoch

        model = get_model(max_len=CFG.max_len, dropout_step=dropout_step, dim=CFG.dim)

        if CFG.transfer_learning:
            for layer in model.layers:
                if layer.name != 'classifier':
                    layer.trainable = False

            model.load_weights(f'{WEIGHTSPATH}/original_weights_best.h5', skip_mismatch=True, by_name=True)

        if CFG.load_weights:
            model.load_weights(f'{WEIGHTSPATH}/original_weights_best.h5', skip_mismatch=True, by_name=True)

        if CFG.export_model:
            config = model.get_config()

            with open('model_config.json', 'w') as f:
                json.dump(config, f)

        # tf.keras.utils.plot_model(model, "lsa_recognition_model.png", show_shapes=True)

        # schedule = OneCycleLR(CFG.lr, CFG.epoch, warmup_epochs=CFG.epoch*CFG.warmup, steps_per_epoch=steps_per_epoch, resume_epoch=CFG.resume, decay_epochs=CFG.epoch, lr_min=CFG.lr_min, decay_type=CFG.decay_type, warmup_type='linear')
        # decay_schedule = OneCycleLR(CFG.lr*CFG.weight_decay, CFG.epoch, warmup_epochs=CFG.epoch*CFG.warmup, steps_per_epoch=steps_per_epoch, resume_epoch=CFG.resume, decay_epochs=CFG.epoch, lr_min=CFG.lr_min*CFG.weight_decay, decay_type=CFG.decay_type, warmup_type='linear')

        awp_step = CFG.awp_start_epoch * steps_per_epoch

        if CFG.fgm:
            model = FGM(model.input, model.output, delta=CFG.awp_lambda, eps=0., start_step=awp_step)
        elif CFG.awp:
            model = AWP(model.input, model.output, delta=CFG.awp_lambda, eps=0., start_step=awp_step)

        # opt = tfa.optimizers.RectifiedAdam(learning_rate=schedule, weight_decay=decay_schedule, sma_threshold=4)#, clipvalue=1.)
        # opt = tfa.optimizers.Lookahead(opt,sync_period=5)

        opt = tf.keras.optimizers.Adam(lr=CFG.lr)

        model.compile(
            optimizer=opt,
            loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)],
            metrics=[tf.keras.metrics.CategoricalAccuracy(),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.AUC(name='auc'),
                     tfa.metrics.F1Score(num_classes=NUM_CLASSES, average='weighted', name='f1_score')
                    ],
            steps_per_execution=steps_per_epoch,
        )

        if summary:
            print()
            model.summary()
            print()
            print(train_ds, valid_ds)
            print()
            # schedule.plot()
            print()
            init = False

        print(f'---------fold{fold}---------')
        print(f'train:{num_train} valid:{num_valid}')
        print()

        if CFG.resume:
            print(f'resume from epoch{CFG.resume}')
            print(f'Fold weights = {CFG.output_dir}/{CFG.comment}-fold{fold}-last.h5')

            model.load_weights(f'{CFG.output_dir}/{CFG.comment}-fold{fold}-last.h5')

            if train_ds is not None:
                model.evaluate(train_ds.take(steps_per_epoch))
            if valid_ds is not None:
                model.evaluate(valid_ds)

        logger = tf.keras.callbacks.CSVLogger(f'{CFG.output_dir}/{CFG.comment}-fold{fold}-logs.csv')

        # It will save the model weights when the validation loss reaches a new minimum value.
        sv_loss = tf.keras.callbacks.ModelCheckpoint(f'{CFG.output_dir}/{CFG.comment}-fold{fold}-best.h5',
                                                     monitor='val_loss', verbose=0, save_best_only=True,
                                                     save_weights_only=True, mode='min', save_freq='epoch')

        snap = Snapshot(f'{CFG.output_dir}/{CFG.comment}-fold{fold}', CFG.snapshot_epochs)

        swa = SWA(f'{CFG.output_dir}/{CFG.comment}-fold{fold}', CFG.swa_epochs, strategy=strategy, train_ds=train_ds,
                  valid_ds=valid_ds, valid_steps=-(num_valid // -CFG.batch_size))
        callbacks = []

        if CFG.save_output:
            callbacks.append(neptune_cbk)
            callbacks.append(logger)
            callbacks.append(snap)
            callbacks.append(swa)

            if fold != 'all':
                callbacks.append(sv_loss)

        history = model.fit(
            train_ds,
            epochs=CFG.epoch - CFG.resume,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_data=valid_ds,
            verbose=CFG.verbose,
            validation_steps=-(num_valid // -CFG.batch_size)
        )

        # Loggear métricas de entrenamiento y validación en Neptune
        run['train/loss'].log(history.history['loss'])
        run['val/loss'].log(history.history['val_loss'])
        run['train/accuracy'].log(history.history['categorical_accuracy'])
        run['val/accuracy'].log(history.history['val_categorical_accuracy'])

        if CFG.save_output:
            try:
                model.load_weights(f'{CFG.output_dir}/{CFG.comment}-fold{fold}-best.weights.h5')
            except:
                pass

        if fold != 'all':
            cv = model.evaluate(valid_ds, verbose=CFG.verbose, steps=-(num_valid // -CFG.batch_size))
        else:
            cv = None

        with open(TRAININGPATH + 'info.csv', 'a', newline='') as outcsv:
            row = [f'{CFG.comment}-fold{fold}-best',
                   f'{CFG.comment}-fold{fold}-last',
                   f'{CFG.comment}-fold{fold}-logs',
                   CFG.n_splits,
                   CFG.seed,
                   CFG.max_len,
                   CFG.replicas,
                   CFG.lr,
                   CFG.epoch,
                   CFG.batch_size,
                   CFG.dim,
                   CFG.transfer_learning,
                   CFG.load_weights]
            writer = csv.writer(outcsv)
            writer.writerow(row)
        
        log_confusion_matrix(model, valid_ds, NUM_CLASSES, name=f'{CFG.comment}-fold{fold}')

        return model, cv, history


def train_folds(CFG, folds, strategy, summary=True):
    """
    CFG: configuration
    folds: list of fold indexes, also can be 'all'.
    In case folds is a list of indexes, those folds will be used as valid files instead of training.
    """
    for fold in folds:
        if fold != 'all':
            all_files = TRAIN_FILENAMES
            train_files = [x for x in all_files if f'fold{fold}' not in x]
            valid_files = [x for x in all_files if f'fold{fold}' in x]
        else:
            train_files = TRAIN_FILENAMES
            valid_files = None

        train_fold(CFG, fold, train_files, strategy, valid_files=valid_files, summary=summary)

    return

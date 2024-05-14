from augmentation import *


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename.split('/')[-1]).group(1)) for filename in filenames]
    return np.sum(n)


for filename in TRAIN_FILENAMES:
    print(filename)

print(count_data_items(TRAIN_FILENAMES), len(train_df))


def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis,
                         keepdims=keepdims) / tf.reduce_sum(
        tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)


def tf_nan_std(x, center=None, axis=0, keepdims=False):
    print("tf_nan_std, center= ", center)
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))


class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks
        print("max_len", self.max_len)
        print("landmarks", self.point_landmarks)

    def call(self, inputs):
        """
             Preprocesses the input data.

             Args:
                 inputs: Input tensor.

             Returns:
                 Preprocessed tensor.
        """
        if tf.rank(inputs) == 3:
            x = inputs[None, ...]
        else:
            x = inputs
        print("Call de preprocess")

        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2)  # N,T,P,C
        std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)

        # Normalization
        x = (x - mean) / std

        # Truncation
        if self.max_len is not None:
            x = x[:, :self.max_len]
        length = tf.shape(x)[1]
        print("1.x shape = ", x)

        x = x[..., :2]
        print("2.x shape = ", x)

        # Velocity calculation
        dx = tf.cond(tf.shape(x)[1] > 1, lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]),
                     lambda: tf.zeros_like(x))
        print("1.dx shape = ", dx)

        dx2 = tf.cond(tf.shape(x)[1] > 2, lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]),
                      lambda: tf.zeros_like(x))

        x = tf.concat([
            tf.reshape(x, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx2, (-1, length, 2 * len(self.point_landmarks))),
        ], axis=-1)

        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)

        return x


def decode_tfrec(record_bytes):
    print("recordbytes: ", record_bytes)
    features = tf.io.parse_single_example(record_bytes, {'coordinates': tf.io.FixedLenFeature([], tf.string),
                                                         'sign': tf.io.FixedLenFeature([], tf.int64), })
    out = {}
    out['coordinates'] = tf.reshape(tf.io.decode_raw(features['coordinates'], tf.float32), (-1, ROWS_PER_FRAME, 3))
    out['sign'] = features['sign']
    print("OUT=", out)
    return out


def filter_nans_tf(x, ref_point=POINT_LANDMARKS):
    mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(tf.gather(x, ref_point, axis=1)), axis=[-2, -1]))
    x = tf.boolean_mask(x, mask, axis=0)
    return x


def preprocess(x, augment=False, max_len=MAX_LEN):
    print("preprocess func")
    print("x['coordinates']: ", x['coordinates'])
    coord = x['coordinates']
    print("XX: ")
    tf.print(x)
    coord = filter_nans_tf(coord)
    if augment:
        coord = augment_fn(coord, max_len=max_len)
    coord = tf.ensure_shape(coord, (None, ROWS_PER_FRAME, 3))

    return tf.cast(Preprocess(max_len=max_len)(coord)[0], tf.float32), tf.one_hot(x['sign'], NUM_CLASSES)


def get_tfrec_dataset(tfrecords, batch_size=64, max_len=64, drop_remainder=False, augment=False, shuffle=False,
                      repeat=False):
    """
    tfrecords: path to tfrecordfiles (which contain lanmarks)
    batch_size
    max_len
    drop_remainder
    augment: temporal and spatial augmentation
    shuffle:
    repeat:

    ver https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
    """
    # Initialize dataset with TFRecords
    ds = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.AUTOTUNE, compression_type='GZIP')
    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    ds = ds.map(lambda x: preprocess(x, augment=augment, max_len=max_len), tf.data.AUTOTUNE)

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(shuffle)
        options = tf.data.Options()
        options.experimental_deterministic = (False)
        ds = ds.with_options(options)

    if batch_size:
        ds = ds.padded_batch(batch_size, padding_values=PAD, padded_shapes=([max_len, CHANNELS], [NUM_CLASSES]),
                             drop_remainder=drop_remainder)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ds = get_tfrec_dataset(TRAIN_FILENAMES, augment=True, batch_size=1024)
ds = get_tfrec_dataset(TRAIN_FILENAMES, augment=False, batch_size=1024)

for x in ds:
    temp_train = x
    break

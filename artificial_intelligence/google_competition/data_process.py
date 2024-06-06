import numpy as np
import tensorflow as tf

from const import MAX_LEN
from const import POINT_LANDMARKS
from const import ROWS_PER_FRAME
from const import NUM_CLASSES
from const import CHANNELS
from const import PAD
from const import LHAND
from const import RHAND
from const import LLIP
from const import RLIP
from const import LPOSE
from const import RPOSE
from const import LEYE
from const import REYE
from const import LNOSE
from const import RNOSE


def tf_nan_mean(x, axis=0, keepdims=False):
    """
    Computes the mean of the input tensor while ignoring NaN values.

    Args:
        x: Input tensor.
        axis: Axis along which to compute the mean. Default is 0.
        keepdims: Whether to keep the dimensions of the input tensor. Default is False.

    Returns:
        The mean of the input tensor with NaN values ignored.
    """
    num = tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims)
    den = tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)
    return num / den


def tf_nan_std(x, center=None, axis=0, keepdims=False):
    """
    Computes the standard deviation of the input tensor while ignoring NaN values.

    Args:
        x: Input tensor.
        center: Tensor representing the mean of the input tensor. If None, the mean is computed internally.
        axis: Axis along which to compute the standard deviation. Default is 0.
        keepdims: Whether to keep the dimensions of the input tensor. Default is False.

    Returns:
        The standard deviation of the input tensor with NaN values ignored.
    """
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))


class Preprocess(tf.keras.layers.Layer):
    """
    Preprocessing layer for input data.

    Args:
        max_len: Maximum length of the input sequence. Default is MAX_LEN from config.
        point_landmarks: List of point landmarks to extract from the input. Default is POINT_LANDMARKS from config.
    """
    def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    @tf.function
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
        x = x[..., :2]

        # Velocity calculation
        dx = tf.cond(tf.shape(x)[1] > 1, lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]),
                     lambda: tf.zeros_like(x))

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
    """
    This function extracts coordinates and sign from TFrecords file
    """
    features = tf.io.parse_single_example(record_bytes, {'coordinates': tf.io.FixedLenFeature([], tf.string),
                                                         'sign': tf.io.FixedLenFeature([], tf.int64), })

    out = {'coordinates': tf.reshape(tf.io.decode_raw(features['coordinates'], tf.float32), (-1, ROWS_PER_FRAME, 3)),
           'sign': features['sign']}
    return out


def filter_nans_tf(x, ref_point=POINT_LANDMARKS):
    """
    x: Tensor with shape = (frames x 543 landmarks x 3 col [x,y,z])
    ref_point: points to extract out of the entire landmarks. In this case are 118 of 543.

    This functions extracts the 118 most important landmarks from the input tensor x. Then checks if a row in Nan.
    If this is True, then it removes it.
    """

    gather = tf.gather(x, ref_point, axis=1)
    nan = tf.math.is_nan(gather)
    reduced = tf.reduce_all(nan, axis=[-2, -1])
    mask = tf.math.logical_not(reduced) # tensor where True indicates that a row has no NaN values in the specified columns
    x = tf.boolean_mask(x, mask, axis=0)
    return x


def preprocess(x, augment=False, max_len=MAX_LEN):
    coord = x['coordinates']
    coord = filter_nans_tf(coord)

    if augment:
        coord = augment_fn(coord, max_len=max_len)
    coord = tf.ensure_shape(coord, (None, ROWS_PER_FRAME, 3))

    preprocessed = Preprocess(max_len=max_len)(coord)[0]
    return tf.cast(preprocessed, tf.float32), tf.one_hot(x['sign'], NUM_CLASSES)


def flip_lr(x):
    """
    This function performs left-right flipping (mirroring)
    """
    x, y, z = tf.unstack(x, axis=-1)
    x = 1 - x  # Inverts the values of x
    new_x = tf.stack([x, y, z], -1)
    new_x = tf.transpose(new_x, [1, 0, 2])  # Transpose the first two dimensions

    # Extracts a sub-tensor from new_x
    lhand = tf.gather(new_x, LHAND, axis=0)
    rhand = tf.gather(new_x, RHAND, axis=0)

    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LHAND)[..., None], rhand)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RHAND)[..., None], lhand)

    llip = tf.gather(new_x, LLIP, axis=0)
    rlip = tf.gather(new_x, RLIP, axis=0)

    # Replaced the channels in new_x corresponding to the left hand (LHAND) with the values from the right
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LLIP)[..., None], rlip)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RLIP)[..., None], llip)

    lpose = tf.gather(new_x, LPOSE, axis=0)
    rpose = tf.gather(new_x, RPOSE, axis=0)

    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LPOSE)[..., None], rpose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RPOSE)[..., None], lpose)

    leye = tf.gather(new_x, LEYE, axis=0)
    reye = tf.gather(new_x, REYE, axis=0)

    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LEYE)[..., None], reye)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(REYE)[..., None], leye)

    lnose = tf.gather(new_x, LNOSE, axis=0)
    rnose = tf.gather(new_x, RNOSE, axis=0)

    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LNOSE)[..., None], rnose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RNOSE)[..., None], lnose)

    new_x = tf.transpose(new_x, [1, 0, 2])

    return new_x


def interp1d_(x, target_len, method='random'):
    """
    Function that helps to randomly select an interpolation algorithm for the tempotral resizing.
    """
    target_len = tf.maximum(1, target_len)  # ensures target length at least of 1
    if method == 'random':
        if tf.random.uniform(()) < 0.33:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bilinear')
        else:
            if tf.random.uniform(()) < 0.5:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bicubic')
            else:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'nearest')
    else:
        x = tf.image.resize(x, (target_len, tf.shape(x)[1]), method)
    return x


def resample(x, rate=(0.8, 1.2)):
    """
    Temporal augmention via resizing with random interpolation algorithms.
    """
    rate = tf.random.uniform((), rate[0], rate[1])
    length = tf.shape(x)[0]  # frames dim
    new_size = tf.cast(rate * tf.cast(length, tf.float32), tf.int32)  # new frame size after resampling
    new_x = interp1d_(x, new_size)

    return new_x


def spatial_random_affine(xyz, scale=(0.8, 1.2), shear=(-0.15, 0.15), shift=(-0.1, 0.1), degree=(-30, 30)):
    """
    Function that performs random spatial transformations: scale, shift, rotate and shear.
    """
    center = tf.constant([0.5, 0.5])  # center point to normalize

    if scale is not None:
        scale = tf.random.uniform((), *scale)
        xyz = scale * xyz

    if shear is not None:
        xy = xyz[..., :2]  # extracts xy
        z = xyz[..., 2:]  # extracts z
        shear_x = shear_y = tf.random.uniform((), *shear)

        # It decides where to apply the shear x or y
        if tf.random.uniform(()) < 0.5:
            shear_x = 0.
        else:
            shear_y = 0.

        shear_mat = tf.identity([
            [1., shear_x],
            [shear_y, 1.]
        ])
        xy = xy @ shear_mat  # tilts the point cloud along the chosen axis (X or Y) based on the random shear values.
        center = center + [shear_y, shear_x]  # updates the center
        xyz = tf.concat([xy, z], axis=-1)

    if degree is not None:
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        xy -= center
        degree = tf.random.uniform((), *degree)
        radian = degree / 180 * np.pi
        c = tf.math.cos(radian)
        s = tf.math.sin(radian)
        rotate_mat = tf.identity([
            [c, s],
            [-s, c],
        ])
        xy = xy @ rotate_mat
        xy = xy + center
        xyz = tf.concat([xy, z], axis=-1)

    if shift is not None:
        shift = tf.random.uniform((), *shift)
        xyz = xyz + shift

    return xyz


def temporal_crop(x, length=MAX_LEN):
    l = tf.shape(x)[0]
    # limits the values in a tensor to a specified minimum and maximum range.
    clipped = tf.clip_by_value(l - length, 1, length)
    offset = tf.random.uniform((), 0, clipped, dtype=tf.int32)
    x = x[offset:offset + length]
    return x


def temporal_mask(x, size=(0.2, 0.4), mask_value=float('nan')):
    """
    Masks portions of the frames.
    """
    l = tf.shape(x)[0]  # frame dimension
    mask_size = tf.random.uniform((), *size)
    mask_size = tf.cast(tf.cast(l, tf.float32) * mask_size, tf.int32)
    mask_offset = tf.random.uniform((), 0, tf.clip_by_value(l - mask_size, 1, l), dtype=tf.int32)
    x = tf.tensor_scatter_nd_update(x, tf.range(mask_offset, mask_offset + mask_size)[..., None],
                                    tf.fill([mask_size, 543, 3], mask_value))
    return x


def spatial_mask(x, size=(0.2, 0.4), mask_value=float('nan')):
    """
    Masks random values of x and y.
    """
    mask_offset_y = tf.random.uniform(())
    mask_offset_x = tf.random.uniform(())
    mask_size = tf.random.uniform((), *size)
    mask_x = (mask_offset_x < x[..., 0]) & (x[..., 0] < mask_offset_x + mask_size)
    mask_y = (mask_offset_y < x[..., 1]) & (x[..., 1] < mask_offset_y + mask_size)
    mask = mask_x & mask_y
    x = tf.where(mask[..., None], mask_value, x)

    return x


def augment_fn(x, always=False, max_len=None):
    """
    Based on random probabilities, this functions selects the temporal and spacial operations
    to apply to the base tensor (mediapipe x, y and z landmarks)
    """
    if tf.random.uniform(()) < 0.8 or always:
        x = resample(x, (0.5, 1.5))
    if tf.random.uniform(()) < 0.5 or always:
        x = flip_lr(x)
    if max_len is not None:
        x = temporal_crop(x, max_len)
    if tf.random.uniform(()) < 0.75 or always:
        x = spatial_random_affine(x)
    if tf.random.uniform(()) < 0.5 or always:
        x = temporal_mask(x)
    if tf.random.uniform(()) < 0.5 or always:
        x = spatial_mask(x)

    return x


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

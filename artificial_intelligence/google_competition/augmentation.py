from const import *


def interp1d_(x, target_len, method='random'):
    print("interp1d")
    length = tf.shape(x)[1]
    target_len = tf.maximum(1, target_len)
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


def flip_lr(x):
    print("flip_lr")
    x, y, z = tf.unstack(x, axis=-1)
    x = 1 - x
    new_x = tf.stack([x, y, z], -1)
    new_x = tf.transpose(new_x, [1, 0, 2])
    lhand = tf.gather(new_x, LHAND, axis=0)
    rhand = tf.gather(new_x, RHAND, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LHAND)[..., None], rhand)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RHAND)[..., None], lhand)
    llip = tf.gather(new_x, LLIP, axis=0)
    rlip = tf.gather(new_x, RLIP, axis=0)
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


def resample(x, rate=(0.8, 1.2)):
    rate = tf.random.uniform((), rate[0], rate[1])
    length = tf.shape(x)[0]
    new_size = tf.cast(rate * tf.cast(length, tf.float32), tf.int32)
    new_x = interp1d_(x, new_size)
    return new_x


def spatial_random_affine(xyz,
                          scale=(0.8, 1.2),
                          shear=(-0.15, 0.15),
                          shift=(-0.1, 0.1),
                          degree=(-30, 30),
                          ):
    center = tf.constant([0.5, 0.5])
    if scale is not None:
        scale = tf.random.uniform((), *scale)
        xyz = scale * xyz

    if shear is not None:
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        shear_x = shear_y = tf.random.uniform((), *shear)
        if tf.random.uniform(()) < 0.5:
            shear_x = 0.
        else:
            shear_y = 0.
        shear_mat = tf.identity([
            [1., shear_x],
            [shear_y, 1.]
        ])
        xy = xy @ shear_mat
        center = center + [shear_y, shear_x]
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
    offset = tf.random.uniform((), 0, tf.clip_by_value(l - length, 1, length), dtype=tf.int32)
    x = x[offset:offset + length]
    return x


def temporal_mask(x, size=(0.2, 0.4), mask_value=float('nan')):
    l = tf.shape(x)[0]
    mask_size = tf.random.uniform((), *size)
    mask_size = tf.cast(tf.cast(l, tf.float32) * mask_size, tf.int32)
    mask_offset = tf.random.uniform((), 0, tf.clip_by_value(l - mask_size, 1, l), dtype=tf.int32)
    x = tf.tensor_scatter_nd_update(x, tf.range(mask_offset, mask_offset + mask_size)[..., None],
                                    tf.fill([mask_size, 543, 3], mask_value))
    return x


def spatial_mask(x, size=(0.2, 0.4), mask_value=float('nan')):
    mask_offset_y = tf.random.uniform(())
    mask_offset_x = tf.random.uniform(())
    mask_size = tf.random.uniform((), *size)
    mask_x = (mask_offset_x < x[..., 0]) & (x[..., 0] < mask_offset_x + mask_size)
    mask_y = (mask_offset_y < x[..., 1]) & (x[..., 1] < mask_offset_y + mask_size)
    mask = mask_x & mask_y
    x = tf.where(mask[..., None], mask_value, x)
    return x


def augment_fn(x, always=False, max_len=None):
    print("augment_fn")
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

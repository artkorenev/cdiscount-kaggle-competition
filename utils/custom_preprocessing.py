from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def preprocessing_factory(preprocessing_name, output_height, output_width, is_training):
    preprocessing_fn = {
        'resize': resize,
        'crop_and_resize': crop_and_resize
    }
    assert preprocessing_name in preprocessing_fn.keys()

    def func(image, label):
        image = preprocessing_fn[preprocessing_name](image, output_height, output_width, is_training)
        return image, label

    return func


def resize(image, output_height, output_width, is_training):
    image = tf.to_float(image)

    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [output_height, output_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)

    R, G, B = tf.split(axis=2, num_or_size_splits=3, value=resized_image)
    R -= _R_MEAN
    G -= _G_MEAN
    B -= _B_MEAN

    return tf.concat(axis=2, values=[R, G, B])


def crop(image, output_height, output_width, is_training):
    if is_training:
        crop = tf.random_crop(image, [output_height, output_width, 3])
    else:
        image_h, image_w, _ = tf.split(tf.shape(image), 3, 0)
        crop = tf.image.crop_to_bounding_box(image,
                                             (tf.squeeze(image_h) - output_height) // 2,
                                             (tf.squeeze(image_w) - output_width) // 2,
                                             output_height, output_width)
    return crop


def crop_and_resize(image, output_height, output_width, is_training):
    crop_ = crop(image, 160, 160, is_training)
    crop_with_resize = resize(crop_, output_height, output_width, is_training)

    return crop_with_resize

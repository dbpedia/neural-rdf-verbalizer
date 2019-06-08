""" Transformer model helper methods """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

_NEG_INF = -1e9

def model_summary(model):
    """
    Gives summary of model and its params
    :param model: the model
    :type model: tf.keras.model object
    :return: summary text
    :rtype: write obj
    """
    model_vars = model.trainable_variables
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_position_encoding(length, hidden_size, min_timescale=1.0,
                            max_timescale=1.0e4):
    """
    Function to get the positional encoding for sequences to
    impart structural information

    :param length: sequence length
    :type length: int
    :param hidden_size: size of hidden state
    :type hidden_size: Tensor
    :param min_timescale:  Minimum scale that will be applied at each position
    :type min_timescale: float
    :param max_timescale: Maximum scale that will be applied at each position
    :type max_timescale: float
    :return: Tensor with shape [length, hidden_size]
    :rtype: tf.Tensor
    """
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

    return signal


def get_decoder_self_attention_bias(length):
    """
    Calculate bias for decoder that maintains model's autoregressive property.
    Creates a tensor that masks out locations that correspond to illegal
    connections, so prediction at position i cannot draw information from future
    positions.
    :param length: length of sequences
    :type length:int
    :return: float tensor of shape [1, 1, length, length]
    :rtype: tf.Tensor
    """
    with tf.name_scope("decoder-self_attention_bias"):
        valid_locs = tf.linalg.band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


def get_padding(x, padding_values=0):
    """
    Returns float tensor representing the padding values
    :param x:  input tensor
    :type x: tf.Tensor
    :param padding_values: padding_value
    :type padding_values: int
    :return: tensor with same shape as x containing values 0 or 1.
    :rtype: tf.tensor
    """
    with tf.name_scope("padding"):
        return tf.cast(tf.equal(x, padding_values), tf.float32)


def get_padding_bias(x):
    """
    Calculate bias tensor from padding values in tensor
    :param x: input tensor
    :type x: tensor
    :return: attention bias
    :rtype: tensor [batch_size, 1, 1, length]
    """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)

        return attention_bias

def loss_function(real, pred, loss_object):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

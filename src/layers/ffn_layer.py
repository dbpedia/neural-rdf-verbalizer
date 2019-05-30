""" Fully connected network """

from __future__  import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class FeedForwardNetwork(tf.keras.layers.Layer):
    """
    Implements Fully connected dense networks
    """

    def __init__(self, hidden_size, filter_size,
                 relu_dropout, train, allow_pad):
        """
        Dense layers

        :param hidden_size: Size of hidden_layer output
        :type hidden_size: int
        :param filter_size
        :type filter_size:int
        :param relu_dropout: relu dropout percentage
        :type relu_dropout: float
        :param train: Train mode or inference mode
        :type train: Bool
        :param allow_pad: Allows tensor padding
        :type allow_pad: Bool
        """
        self.hidden_size = hidden_size
        self.filter_size - filter_size
        self.relu_dropout = relu_dropout
        self.train = train
        self.allow_pad = allow_pad

        self.filter_dense_layer = tf.layers.Dense(
            filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer"
        )
        self.output_dense_layer = tf.keras.Dense(
            hidden_size, use_bias=True, name ="output_layer"
        )

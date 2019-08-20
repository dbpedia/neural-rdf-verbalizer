"""Implementation of fully connected network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FeedForwardNetwork(tf.keras.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(self, hidden_size, filter_size, relu_dropout):
        """Initialize FeedForwardNetwork.

        Args:
          hidden_size: int, output dim of hidden layer.
          filter_size: int, filter size for the inner (first) dense layer.
          relu_dropout: float, dropout rate for training.
        """
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout

    def build(self, input_shape):
        self.filter_dense_layer = tf.keras.layers.Dense(
            self.filter_size,
            use_bias=True,
            activation=tf.nn.relu,
            name="filter_layer")
        self.output_dense_layer = tf.keras.layers.Dense(
            self.hidden_size, use_bias=True, name="output_layer")
        super(FeedForwardNetwork, self).build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "filter_size": self.filter_size,
            "relu_dropout": self.relu_dropout,
        }

    def call(self, x, training):
        """Return outputs of the feedforward network.

        Args:
          x: tensor with shape [batch_size, length, hidden_size]
          training: boolean, whether in training mode or not.

        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, length, hidden_size]
        """
        # Retrieve dynamically known shapes
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        output = self.filter_dense_layer(x)
        if training:
            output = tf.nn.dropout(output, rate=self.relu_dropout)
        output = self.output_dense_layer(output)

        return output

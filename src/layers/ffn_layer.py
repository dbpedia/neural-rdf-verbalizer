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
        self.output_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=True, name ="output_layer"
        )

    def call(self, x, padding=None):
        """
        Passes tensor through Feed Forward networks
        :param x: input tensor
        :type x: tf.Tensor
        :param padding: If set padding values are removed temporarily
        :type padding: Boolean
        :return: Outputs of FFN
        :rtype: tf.Tensor
        """
        padding = None if not self.allow_pad else padding
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        if padding is not None:
            with tf.name_scope("remove_padding"):
                pad_mask = tf.reshape(padding, [-1])
                nonpad_ids = tf.to_int32(tf.where(pad_mask<1e-9))

            #reshape x
            x = tf.reshape(x, [-1, self.hidden_size])
            x = tf.gather_nd(x, indices=nonpad_ids)

            # Reshape x from 2 dimensions to 3 dimensions.
            x.set_shape([None, self.hidden_size])
            x = tf.expand_dims(x, axis=0)

        output = self.filter_dense_layer(x)
        if self.train:
            output = tf.nn.dropout(output, 1.0 - self.relu_dropout)
        output = self.output_dense_layer(output)

        if padding is not None:
            with tf.name_scope("re_add_padding"):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(
                    indices=nonpad_ids,
                    updates = output,
                    shape=[batch_size*length, self.hidden_size]
                )

                output = tf.reshape(output, [batch_size, length, self.hidden_size])

            return output



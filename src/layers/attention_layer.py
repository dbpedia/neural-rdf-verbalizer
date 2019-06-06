""" Implementaiton of Multiheaded Attention Layers """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    """ MultiHeaded Attention Layer """

    def __init__(self, hidden_size, num_heads, attention_dropout):
        """

        :param hidden_size: output dim of hidden layer
        :type hidden_size: int
        :param num_heads: num of heads to repeat the attention mechanism
        :type num_heads: int
        :param attention_dropout: dropout probability for attention layers
        :type attention_dropout: float
        """

        if hidden_size % num_heads:
            raise ValueError(
                "Hidden Size muse be divisible by number of heads"
            )

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

    def build(self, input_shape):
        """
        Builds the attention layer

        :param input_shape:
        :type tensor : input tensor shape
        :return:
        :rtype:
        """
        # q, k and v layers
        self.q_dense_layer = tf.keras.layers.Dense(
            self.hidden_size, use_bias=False, name="q"
        )
        self.k_dense_layer = tf.keras.layers.Dense(
            self.hidden_size, use_bias=False, name="k"
        )
        self.v_dense_layer = tf.keras.layers.Dense(
            self.hidden_size, use_bias=False, name="v"
        )
        self.output_dense_layer = tf.layers.Dense(
            self.hidden_size, use_bias=False, name="output_transform"
        )

        super(Attention, self).build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
        }

    def split_heads(self, x):
        """
        Split x into different heads, and transpose the result

        :param x: input tensor [batch, length, hidden_size]
        :type x: tensor
        :return: tensor [batch, length, num_heads, hidden_size // num_heads]
        :rtype: tensor
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            #calculate the depth
            depth = (self.hidden_size // self.num_heads)

            x = tf.reshape(x, [batch_size,length,self.num_heads,depth])

            return tf.transpose(x, [0,2,1,3])

    def combine_heads(self, x):
        """
        Combine the tensor that was split

        :param x: input tensor [batch_size, num_heads, length, depth
        :type x: tensor
        :return: tensor [batch_size, length, hidden_size]
        :rtype: tensor
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0,2,1,3])

            return tf.reshape(x, [batch_size,length,self.hidden_size])

    def call(self, x, y, bias, training, cache=None):
        """
        Apply attention to x and y

        :param x: tensor with shape [batch_size, length, hidden_size]
        :type x:  tensor
        :param y: tensor with shape same as x
        :type y:  tensor
        :param bias: attention bias that will be added to result of dot product
        :type bias:  tensor
        :param training: Training mode or not
        :type training: Boolean
        :param cache: dict with tensors of results
        :type cache: dict {"k": tensor
                            "v": tensor
                            }
        :return: attention layer outputs [batch_size, length, hidden_size]
        :rtype: tensor
        """
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        if training:
            weights = tf.nn.dropout(weights, rate=self.attention_dropout)
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output

class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, training, cache=None):
        return super(SelfAttention, self).call(x, x, bias, training, cache)

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

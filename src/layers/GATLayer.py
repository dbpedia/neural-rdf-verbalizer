""" Graph Attention Network layer """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.enable_eager_execution()


class GraphAttentionLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, num_heads, reg_scale=0.001, rate=0.1):
    """
    Graph Attention Network Layer, takes input and returns embedded
    node features with self attention applied on the feature matrix
    """
    super(GraphAttentionLayer, self).__init__()
    self.in_dim = d_model
    self.out_dim = dff
    self.num_heads = num_heads
    self.dropout_rate = rate
    self.kernels = []
    self.biases = []
    self.attn_kernels = []

    self.edge_layer = tf.keras.layers.Dense(self.out_dim)
    self.lrelu = tf.keras.layers.LeakyReLU()
    self.dropout = tf.keras.layers.Dropout(rate)
    self.reg = tf.contrib.layers.l2_regularizer(reg_scale)

    for head in range(self.num_heads):
      kernel = self.add_weight(shape=(self.in_dim, self.out_dim),
                               initializer='glorot_uniform',
                               regularizer=self.reg,
                               name='kernel_{}'.format(head))
      bias = self.add_weight(shape=(self.out_dim,),
                             initializer='glorot_uniform',
                             regularizer=self.reg,
                             name='bias_{}'.format(head))
      self.kernels.append([kernel, bias])
      # Attention kernels
      attn_kernel_self = self.add_weight(shape=(self.out_dim, 1),
                                         initializer='glorot_uniform',
                                         regularizer=self.reg,
                                         name='attn_kernel_self_{}'.format(head))
      attn_kernel_neighs = self.add_weight(shape=(self.out_dim, 1),
                                           initializer='glorot_uniform',
                                           regularizer=self.reg,
                                           name='attn_kernel_neigh_{}'.format(head))
      self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

  def call(self, nodes, edges, labels, num_heads, training, mask=None):
    edges = self.edge_layer(tf.add(edges, labels))
    inputs = nodes

    outputs = []
    for head in range(num_heads):
      kernel = self.kernels[head]
      attention_kernel = self.attn_kernels[head]
      features = tf.keras.backend.dot(inputs, kernel[0])
      features = tf.add(features, kernel[1])
      attn_for_self = tf.keras.backend.dot(features, attention_kernel[0])
      attn_for_neighs = tf.keras.backend.dot(features, attention_kernel[1])
      # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
      features = tf.add(features, edges)
      dense = tf.matmul(attn_for_self, attn_for_neighs, transpose_b=True)
      dense = self.lrelu(dense)

      # Mask values before activation (Vaswani et al., 2017)
      # mask_local = -10e9 * (1.0 - adj)
      # dense += mask_local

      # Apply softmax to get attention coefficients
      dense = tf.math.softmax(dense)  # (N x N)

      # Apply dropout to features and attention coefficients
      if training is True:
        dense = self.dropout(dense)  # (N x N)
        features = self.dropout(features)  # (N x F')

      # Linear combination with neighbors' features
      node_features = tf.matmul(dense, features)  # (N x F')
      outputs.append(node_features)

    output = tf.reduce_mean(tf.stack(outputs), axis=0)  # N x F')
    output = tf.nn.relu(output)

    return output

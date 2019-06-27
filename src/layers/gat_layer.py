""" Graph Attention Network layer """

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from src.layers.attention_layer import SelfAttention, MultiHeadAttention
from src.utils.model_utils import point_wise_feed_forward_network

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

class GraphAttentionLayer (tf.keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, rate=0.1):
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

        self.node_layer = tf.keras.layers.Dense(dff)
        self.edge_layer = tf.keras.layers.Dense(dff)
        self.lrelu = tf.keras.layers.LeakyReLU()
        self.dropout = tf.keras.layers.Dropout(rate)

        for head in range(self.num_heads):
            kernel = self.add_weight(shape=(self.in_dim, self.out_dim),
                                     initializer='glorot_uniform',
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)
            bias = self.add_weight(shape=(self.out_dim, ),
                                   initializer='glorot_uniform',
                                   name='bias_{}'.format(head))
            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.out_dim, 1),
                                               initializer='glorot_uniform',
                                               name='attn_kernel_self_{}'.format(head))
            attn_kernel_neighs = self.add_weight(shape=(self.out_dim, 1),
                                                 initializer='glorot_uniform',
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

    def call(self, nodes, adj, num_heads, training, mask=None):
        nodes = self.node_layer(nodes)
        #edges = self.edge_layer(edges)
        #inputs = tf.add(nodes, edges)
        inputs = nodes

        outputs = []
        for head in range(num_heads):
            kernel = self.kernels[head]
            attention_kernel = self.attn_kernels[head]

            features = tf.keras.backend.dot(inputs, kernel)
            attn_for_self = tf.keras.backend.dot(features, attention_kernel[0])
            attn_for_neighs = tf.keras.backend.dot(features, attention_kernel[1])
            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]

            dense = tf.add(attn_for_neighs, attn_for_self)
            dense = tf.broadcast_to(dense, shape=adj.shape)
            dense = self.lrelu(dense)

            # Mask values before activation (Vaswani et al., 2017)
            mask_local = -10e9 * (1.0 - adj)
            dense += mask_local

            # Apply softmax to get attention coefficients
            dense = tf.math.softmax(dense)  # (N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = self.dropout(dense)  # (N x N)
            dropout_feat = self.dropout(features)  # (N x F')

            # Linear combination with neighbors' features
            node_features = tf.keras.backend.dot(dropout_attn, dropout_feat)  # (N x F')
            outputs.append(node_features)

        output = tf.reduce_mean(tf.stack(outputs), axis=0)  # N x F')
        output = tf.nn.relu(output)

        return output
""" Graph Attention Network layer """

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from src.layers.attention_layer import SelfAttention, MultiHeadAttention
from src.utils.model_utils import point_wise_feed_forward_network

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, rate=0.1):
    """
    Graph Attention Network Layer, takes input and returns embedded
    node features with self attention applied on the feature matrix
    """
        super(GraphAttentionLayer, self).__init__()

        self.mha = MultiHeadAttention(dff, num_heads)
        self.ffn = point_wise_feed_forward_network(dff, dff)
        self.layernorm1 = tf.contrib.layers.layer_norm
        self.layernorm2 = tf.contrib.layers.layer_norm

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.node_layer = tf.keras.layers.Dense(dff)
        self.edge_layer = tf.keras.layers.Dense(dff)

    def call(self, nodes, edges, adj, num_heads, training, mask=None):

        nodes = self.node_layer(nodes)
        edges = self.edge_layer(edges)
        input = tf.add(nodes, edges)
        x = tf.matmul(adj, input)

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        
        return out2
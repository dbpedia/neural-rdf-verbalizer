""" Graph Attention Network layer """

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from src.layers.attention_layer import SelfAttention

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

class GraphAttentionLayer(tf.keras.layers.Layer):
    """
    Graph Attention Network Layer, takes input and returns embedded
    node features with self attention applied on the feature matrix
    """
    def __init__(self, in_dim, out_dim, num_heads, alpha, dropout=0.2,
                 bias=False):
        """
        Initialises Graph Attention Layer
        :param in_dim: input dimensions
        :type in_dim: int
        :param out_dim: Output vector dimensions
        :type out_dim: int
        :param dropout: dropout probability
        :type dropout: float
        :param bias: add bias or not
        :type bias: Bool
        :param edges: add edge features or not
        :type edges: Bool
        :param train: in training mode or eval mode
        :type train: Bool
        """
        super(GraphAttentionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.bias = bias
        self.num_heads = num_heads

        self.w1_layer = tf.keras.layers.Dense(
            self.out_dim, use_bias=True, name="Weights_1", kernel_initializer='glorot_normal',
            bias_initializer='zeros'
        )
        self.w2_layer = tf.keras.layers.Dense(
            self.out_dim, use_bias=True, name="weights_2", kernel_initializer='glorot_normal',
            bias_initializer='zeros'
        )
        self.self_attention = SelfAttention(out_dim, num_heads, self.dropout)
        self.dense = tf.keras.layers.Dense
        self.multi_head_dense = tf.keras.layers.Dense
        self.Dropout = tf.keras.layers.Dropout(
            self.dropout
        )
        self.lrelu = tf.keras.layers.LeakyReLU(alpha)
        self.layernorm1 = tf.contrib.layers.layer_norm
        self.layernorm2 = tf.contrib.layers.layer_norm
        self.layernorm3 = tf.contrib.layers.layer_norm


    def __call__(self, inputs, adj, num_heads, train):
        """
        Propagates the adjacency matrix and node feature matrix through
        the layer and calculates the attention coefficients.

        Follows the propogation rule
        h' = W*inputs 
        h' = h'*(A + I) 
        eij =  softmax(h') 
        This makes sure the features of a node are sum of all first order neighbour's 
        features and it's own features 
        
        :param inputs: node feature matrix 
        :type inputs: tf.tensor  [batchsize, nodes, in_features]
        :param adj:adjacency matrix 
        :type adj:tf.tensor 
        :return: encoded node representations 
        :rtype:tf.tensor 
        """
        """
        self.num_heads = num_heads
        batch_size = inputs.get_shape().as_list()[0]
        nodes = adj.get_shape().as_list()[1]
        inputs = tf.matmul(adj, inputs)  #[batch_size, nodes, in_dim]

        hidden_state = self.w1_layer(inputs) #[batch_size, nodes, out_dim]
        if train == True:
            hidden_state = self.Dropout(hidden_state)

        hidden_state = self.lrelu(hidden_state)
        hidden_state = self.w2_layer(hidden_state)
        if train == True:
            hidden_state = self.Dropout(hidden_state)

        hidden_state = self.lrelu(hidden_state)
        #Apply attention mechanism now

        output = self.self_attention(hidden_state, bias=False, training=False)
        output = self.layernorm3(output)
        """
        self.num_heads = num_heads
        batch_size = inputs.get_shape().as_list()[0]
        nodes = adj.get_shape().as_list()[1]

       # outputs = self.w1_layer(inputs)
       # outputs = self.layernorm1(outputs)

        coef = self.dense(nodes)(inputs)
        coef = tf.math.softmax(coef)

        adj = tf.math.multiply(adj, coef)
        output = tf.matmul(adj, inputs)
        output = self.layernorm2(output)
        output = self.lrelu(output)

        return output


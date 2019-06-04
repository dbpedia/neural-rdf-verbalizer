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
    def __init__(self, in_dim, out_dim, num_heads,dropout=0.2,
                 bias=False, edges=True, train=True):
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

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.bias = bias
        self.edges = edges
        self.train = train
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


    def __call__(self, inputs, adj, num_heads):
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
        self.num_heads = num_heads
        batch_size = inputs.get_shape().as_list()[0]
        nodes = adj.get_shape().as_list()[1]
        inputs = tf.matmul(adj, inputs)  #[batch_size, nodes, in_dim]
         
        hidden_state = self.w1_layer(inputs) #[batch_size, nodes, out_dim]
        hidden_state = self.w2_layer(hidden_state)
        #Apply attention mechanism now

        output = self.self_attention(hidden_state, bias=False, training=False)

        return output







        



         
        
        



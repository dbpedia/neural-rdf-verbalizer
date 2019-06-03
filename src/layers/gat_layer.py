""" Graph Attention Network layer """

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

class GraphAttentionLayer(tf.keras.layers.Layer):
    """
    Graph Attention Network Layer, takes input and returns embedded
    node features with self attention applied on the feature matrix
    """
    def __init__(self, in_dim, out_dim, dropout=0.2,
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
        in_dim = self.in_dim
        out_dim = self.out_dim
        dropout = self.dropout
        bias = self.bias
        edges = self.edges
        train = self.train

        #Initialise q, k, and v weights 
        wQ = tf.get_variable("Q_matrix", shape=[])

    def call(self, inputs, adj, num_heads):
        """
        Propagates the adjacency matrix and node feature matrix through
        the layer and calculates the attention coefficients.

        Follows the propogation rule
        h' = W*inputs 
        h' = h'*(A + I) 
        eij =  softmax(h') 
        
        :param inputs: node feature matrix 
        :type inputs: tf.tensor 
        :param adj:adjacency matrix 
        :type adj:tf.tensor 
        :return: encoded node representations 
        :rtype:tf.tensor 
        """
        nodes = adj.get_shape().as_list()[0]
        identity = tf.eye(nodes, dtype=tf.float32) 
        adj = tf.add(adj, identity) #add self-loops to the adjascency matrix 

        W = tf.get_variable("weights", shape=[self.input_dim, self.output_dim],
                            dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        inputs = tf.matmul(adj, inputs) 
        hidden_state = tf.matmul(inputs, W) 
        



         
        
        



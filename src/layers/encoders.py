""" Base and encoder classes """

from __future__ import  division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from src.layers.gat_layer import GraphAttentionLayer
import abc
from collections import namedtuple
import six

class GraphEncoder(tf.keras.layers.Layer):
    """
    Class the defines and initializes graph Encoder stack 
    """
    def __init__(self, args, train):
        super(GraphEncoder, self).__init__()
        in_dim = args.emb_dim
        out_dim = args.hidden_size
        num_heads = args.num_heads
        self.num_heads = num_heads
        dropout = args.dropout 
        bias = args.use_bias
        edges = args.use_edges
        num_layers = args.num_layers  
        self.layers = [] 

        for i in range(num_layers):
            if i==0:
                gat_layer = GraphAttentionLayer(in_dim, out_dim, num_heads,
                                                dropout, bias, edges, train)
            else:
                gat_layer = GraphAttentionLayer(out_dim, out_dim, num_heads,
                                                dropout, bias, edges, train) 
            self.layers.append(gat_layer)

    def __call__(self, inputs, adj):
        with tf.variable_scope("encoding"):
            for i in range(self.num_heads):
                if i==0:
                    outputs = self.layers[i](inputs, adj, self.num_heads)
                else:
                    # Skip connections
                    shortcut = outputs
                    outputs = self.layers[i](outputs, adj, self.num_heads)
                    outputs += shortcut

        return outputs


    

            
















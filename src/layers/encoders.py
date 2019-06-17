""" Base and encoder classes """

from __future__ import  division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from src.layers.gat_layer import GraphAttentionLayer
from src.layers.attention_layer import MultiHeadAttention
from src.utils.model_utils import point_wise_feed_forward_network
import abc
from collections import namedtuple
import six

class GraphEncoder(tf.keras.layers.Layer):
    """
    Class the defines and initializes graph Encoder stack 
    """
    def __init__(self, args):
        super(GraphEncoder, self).__init__()
        in_dim = args.emb_dim
        out_dim = args.hidden_size
        num_heads = args.num_heads
        self.num_heads = num_heads
        dropout = args.dropout 
        bias = args.use_bias
        edges = args.use_edges
        num_layers = args.num_layers
        units = args.enc_units
        alpha = args.alpha

        self.layers = []

        for i in range(num_layers):
            if i==0:
                gat_layer = GraphAttentionLayer(in_dim, out_dim, num_heads,
                                                alpha, dropout, bias)
                self.layers.append(gat_layer)
            else:

                dense_layer = tf.keras.layers.Dense(out_dim)
                self.layers.append(dense_layer)

        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True, recurrent_initializer='glorot_uniform')
        self.hidden = tf.zeros((args.batch_size, args.enc_units))



    def __call__(self, inputs, adj, train):
        with tf.variable_scope("encoding"):
            for i in range(self.num_heads):
                if i==0:
                    outputs = self.layers[i](inputs, adj, self.num_heads, train)
                else:
                    # Skip connections
                    shortcut = outputs
                    #outputs = self.graph_layers[i](outputs, adj, self.num_heads, train)
                    outputs = self.layers[i](outputs)
                    outputs += shortcut
            outputs, state = self.gru(outputs, initial_state=self.hidden)
        return outputs, state

class RNNEncoder(tf.keras.layers.Layer):
    """
    RNN Encoder
    """
    def __init__(self, vocab_size, emb_dim, enc_units, batch_size):
        """

        :param args: All Arguments given to the model
        :type args: argparse object
        """
        super(RNNEncoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.contrib.layers.layer_norm
        self.layernorm2 = tf.contrib.layers.layer_norm

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

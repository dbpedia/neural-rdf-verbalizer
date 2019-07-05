""" Base and encoder classes """

from __future__ import  division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from src.layers.gat_layer import GraphAttentionLayer
from src.layers.attention_layer import MultiHeadAttention
from src.utils.model_utils import point_wise_feed_forward_network, positional_encoding

class GraphEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, node_vocab_size, role_vocab_size,
                 reg_scale=0.001, rate=0.1):
      
        super(GraphEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.node_embedding = tf.keras.layers.Embedding(node_vocab_size, d_model)
        # 4 = subject, object, predicate, bridge
        self.role_embedding = tf.keras.layers.Embedding(role_vocab_size, d_model)
        self.node_pos_encoding = positional_encoding(node_vocab_size, self.d_model)
        self.node_role_layer = tf.keras.layers.Dense(self.d_model, input_shape=(2*d_model, ))

        self.enc_layers = [GraphAttentionLayer(d_model, dff, num_heads,
                                               reg_scale=reg_scale, rate=rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        self.layernorm = tf.contrib.layers.layer_norm

    def call(self, nodes, adj, roles, num_heads, training, mask):
        node_seq_len = tf.shape(nodes)[1]

        # adding embedding and position encoding.
        node_tensor = self.node_embedding(nodes)  # (batch_size, input_seq_len, d_model)
        adj = tf.cast(adj, dtype=tf.float32)

        node_tensor *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        node_tensor += self.node_pos_encoding[:, :node_seq_len, :]

        role_tensor = self.role_embedding(roles)  # (batch_size, input_seq_len, d_model)
        role_tensor *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        role_tensor += self.node_pos_encoding[:, :node_seq_len, :]

        node_tensor = tf.concat([node_tensor, role_tensor], 2)
        node_tensor = tf.nn.relu(self.node_role_layer(node_tensor))
        node_tensor = self.dropout(node_tensor, training=training)

        for i in range(self.num_layers):
            if i==0:
                x = self.enc_layers[i](node_tensor, adj, num_heads, training, mask)
            elif((i % 2)==0):
                shortcut = x
                x = self.enc_layers[i](node_tensor, adj, num_heads, training, mask)
                x += shortcut
            else:
                x = self.enc_layers[i](node_tensor, adj, num_heads, training, mask)

        return self.layernorm(x)  # (batch_size, input_seq_len, d_model)

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
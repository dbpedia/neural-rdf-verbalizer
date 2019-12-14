""" Base and encoder classes """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src.layers.AttentionLayer import MultiHeadAttention
from src.layers.GATLayer import GraphAttentionLayer
from src.layers.ffn_layer import FeedForwardNetwork
from src.utils.model_utils import point_wise_feed_forward_network


class GraphEncoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff,
               filter_size, reg_scale=0.001, rate=0.1):

    super(GraphEncoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers

    self.node_role_layer = tf.keras.layers.Dense(self.d_model, input_shape=(2 * d_model,))
    self.enc_layers = []
    for _ in range(num_layers):
      gat_layer = GraphAttentionLayer(d_model, dff, num_heads,
                                      reg_scale=reg_scale, rate=rate)
      ffn_layer = FeedForwardNetwork(dff, filter_size, rate)
      self.enc_layers.append([gat_layer, ffn_layer])

    self.dropout = tf.keras.layers.Dropout(rate)
    self.layernorm = tf.contrib.layers.layer_norm

  def call(self, node_tensor, label_tensor, node1_tensor, node2_tensor, num_heads, training):
    # adding embedding and position encoding.

    edge_tensor = tf.concat([node1_tensor, node2_tensor], 2)
    edge_tensor = tf.cast(self.node_role_layer(edge_tensor), dtype=tf.float32)
    # node_tensor = tf.add(node_tensor, role_tensor)
    node_tensor *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    edge_tensor *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    # node_tensor += self.node_pos_enc[:, :node_seq_len, :]

    for i, layer in enumerate(self.enc_layers):
      if i == 0:
        x = self.enc_layers[i][0](node_tensor, edge_tensor, label_tensor, num_heads, training)
        x = self.enc_layers[i][1](x, training=self.trainable)
      else:
        shortcut = x
        x = self.enc_layers[i][0](node_tensor, edge_tensor, label_tensor, num_heads, training)
        x = self.enc_layers[i][1](x, training=self.trainable)
        x += shortcut

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
    self.forward_gru = tf.keras.layers.CuDNNGRU(self.enc_units,
                                                return_sequences=True,
                                                return_state=True,
                                                recurrent_initializer='glorot_uniform')
    self.backward_gru = tf.keras.layers.CuDNNGRU(self.enc_units,
                                                 return_sequences=True,
                                                 return_state=True,
                                                 go_backwards=True,
                                                 recurrent_initializer='glorot_uniform')
    self.gru = tf.keras.layers.Bidirectional(self.forward_gru, backward_layer=self.backward_gru,
                                             merge_mode='ave')

  def __call__(self, x, hidden):
    x = self.embedding(x)
    output = self.gru(x, initial_state=hidden)
    output, state = output[0], output[2]
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

"""
        Pure - RNN model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src.layers.Decoders import RNNDecoder
from src.layers.Encoders import RNNEncoder
from src.utils.model_utils import loss_function


class RNNModel(tf.keras.Model):
  """
  RNN model
  """

  def __init__(self, vocab_inp_size, vocab_tgt_size, target_lang, args):
    super(RNNModel, self).__init__()
    self.vocab_inp_size = vocab_inp_size
    self.vocab_tgt_size = vocab_tgt_size
    self.args = args
    self.batch_size = args.batch_size
    self.target_lang = target_lang
    self.encoder = RNNEncoder(self.vocab_inp_size, args.emb_dim,
                              args.enc_units, args.batch_size)
    self.decoder = RNNDecoder(self.vocab_tgt_size, args.emb_dim,
                              args.enc_units, args.batch_size)
    self.loss_object = tf.keras.losses.sparse_categorical_crossentropy

  def __call__(self, inp, targ, enc_hidden):
    loss = 0
    enc_output, enc_hidden = self.encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([self.target_lang.word_index['<start>']] * self.batch_size, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions, self.loss_object)
      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

    return predictions, dec_hidden, loss

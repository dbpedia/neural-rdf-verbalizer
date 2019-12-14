"""     Defines the mgraph attention encoder model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from src.layers import EmbeddingLayer
from src.layers.Decoders import RNNDecoder
from src.layers.Encoders import GraphEncoder
from src.models.Transformer import DecoderStack
from src.utils import TransformerUtils
from src.utils import beam_search
from src.utils.metrics import MetricLayer
from src.utils.model_utils import loss_function


class GATModel(tf.keras.Model):
  """
  Model that uses Graph Attention encoder and RNN decoder (for now)
  """

  def __init__(self, args, src_vocab_size, tgt_vocab_size, target_lang):
    super(GATModel, self).__init__()
    self.emb_layer = EmbeddingLayer.EmbeddingSharedWeights(
      src_vocab_size, args.emb_dim)

    self.tgt_emb_layer = EmbeddingLayer.EmbeddingSharedWeights(
      tgt_vocab_size, args.emb_dim)

    self.encoder = GraphEncoder(args.enc_layers, args.emb_dim, args.num_heads, args.hidden_size,
                                args.filter_size, reg_scale=args.reg_scale, rate=args.dropout)
    self.decoder = RNNDecoder(tgt_vocab_size, args.emb_dim, args.enc_units, args.batch_size)
    self.vocab_tgt_size = tgt_vocab_size
    self.num_heads = args.num_heads
    self.target_lang = target_lang
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    self.hidden = tf.keras.layers.Dense(args.hidden_size)

  def __call__(self, nodes, labels, node1, node2, targ):
    """
    Puts the tensors through encoders and decoders
    :param adj: Adjacency matrices of input example
    :type adj: tf.tensor
    :param nodes: node features
    :type nodes: tf.tensor
    :param targ: target sequences
    :type targ: tf.tensor
    :return: output probability distribution
    :rtype: tf.tensor
    """
    node_tensor = tf.cast(self.emb_layer(nodes), dtype=tf.flaot32)
    label_tensor = tf.cast(self.emb_layer(labels), dtype=tf.float32)
    node1_tensor = tf.cast(self.emb_layer(node1), dtype=tf.float32)
    node2_tensor = tf.cast(self.emb_layer(node2), dtype=tf.float32)

    enc_output = self.encoder(node_tensor, label_tensor, node1_tensor, node2_tensor,
                              self.num_heads, self.encoder.trainable)
    batch = enc_output.shape[0]
    self.enc_output_hidden = tf.reshape(enc_output, shape=[batch, -1])
    enc_hidden = self.hidden(self.enc_output_hidden)

    dec_input = tf.expand_dims([self.target_lang.word_index['<start>']] * self.args.batch_size, 1)
    loss = 0

    for t in range(1, targ.shape[1]):
      # pass encoder output to decoder
      predictions, dec_hidden, _ = self.decoder(dec_input, enc_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions, self.loss_object)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

    return predictions, dec_hidden, loss


class TransGAT(tf.keras.Model):
  """
  Model that uses Graph Attention encoder and RNN decoder (for now)
  """

  def __init__(self, args, src_vocab_size, src_lang,
               tgt_vocab_size, max_seq_len, tgt_vocab):
    super(TransGAT, self).__init__()
    self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    self.emb_layer = EmbeddingLayer.EmbeddingSharedWeights(
      src_vocab_size, args.emb_dim)

    self.tgt_emb_layer = EmbeddingLayer.EmbeddingSharedWeights(
      tgt_vocab_size, args.emb_dim)

    self.metric_layer = MetricLayer(tgt_vocab_size)

    self.encoder = GraphEncoder(args.enc_layers, args.emb_dim, args.num_heads, args.hidden_size,
                                args.filter_size, reg_scale=args.reg_scale, rate=args.dropout)
    self.decoder_stack = DecoderStack(args)
    self.vocab_tgt_size = tgt_vocab_size
    self.target_lang = src_lang
    self.args = args
    self.num_heads = args.num_heads
    self.max_len = max_seq_len
    if self.args.distillation == 'True':
      self.temp = tf.constant(self.args.temp
                              , dtype=tf.float32)
    else:
      self.temp = tf.constant(1, dtype=tf.float32)

  def _get_symbols_to_logits_fn(self, max_decode_length, training):
    """Returns a decoding function that calculates logits of the next tokens."""

    timing_signal = TransformerUtils.get_position_encoding(
      max_decode_length + 1, self.args.emb_dim)
    decoder_self_attention_bias = TransformerUtils.get_decoder_self_attention_bias(
      max_decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.
      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size *
          beam_size, i + 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.
      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.tgt_emb_layer(decoder_input)
      decoder_input += timing_signal[i:i + 1]

      self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
      decoder_outputs = self.decoder_stack(
        decoder_input,
        cache.get("encoder_outputs"),
        self_attention_bias,
        cache.get("encoder_decoder_attention_bias"),
        training=training,
        cache=cache)
      logits = self.tgt_emb_layer(decoder_outputs, mode="linear")
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return symbols_to_logits_fn

  def predict(self, encoder_outputs, encoder_decoder_attention_bias, training):
    """Return predicted sequence."""
    encoder_outputs = tf.cast(encoder_outputs, tf.float32)
    batch_size = tf.shape(encoder_outputs)[0]
    input_length = tf.shape(encoder_outputs)[1]
    max_decode_length = self.max_len

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(
      max_decode_length, training)
    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)
    cache = {
      "layer_%d" % layer: {
        "k": tf.zeros([batch_size, 0, self.args.hidden_size]),
        "v": tf.zeros([batch_size, 0, self.args.hidden_size])
      } for layer in range(self.args.enc_layers)
    }
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = beam_search.sequence_beam_search(
      symbols_to_logits_fn=symbols_to_logits_fn,
      initial_ids=initial_ids,
      initial_cache=cache,
      vocab_size=self.vocab_tgt_size,
      beam_size=self.args.beam_size,
      alpha=self.args.beam_alpha,
      max_decode_length=max_decode_length,
      eos_id=6)

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]
    return {"outputs": top_decoded_ids, "scores": top_scores}

  def __call__(self, nodes, labels, node1, node2, targ, mask):
    """
    Puts the tensors through encoders and decoders
    :param adj: Adjacency matrices of input example
    :type adj: tf.tensor
    :param nodes: node features
    :type nodes: tf.tensor
    :param targ: target sequences
    :type targ: tf.tensor
    :return: output probability distribution
    :rtype: tf.tensor
    """
    node_tensor = self.emb_layer(nodes)
    label_tensor = tf.cast(self.emb_layer(labels), dtype=tf.float32)
    node1_tensor = tf.cast(self.emb_layer(node1), dtype=tf.float32)
    node2_tensor = tf.cast(self.emb_layer(node2), dtype=tf.float32)

    enc_output = self.encoder(node_tensor, label_tensor, node1_tensor, node2_tensor,
                              self.num_heads, self.encoder.trainable)
    attention_bias = TransformerUtils.get_padding_bias(nodes)
    attention_bias = tf.cast(attention_bias, tf.float32)

    if targ is not None:
      decoder_inputs = tf.cast(self.tgt_emb_layer(targ), dtype=tf.float32)
    else:
      predictions = self.predict(enc_output, attention_bias, False)
      return predictions

    with tf.name_scope("shift_targets"):
      # Shift targets to the right, and remove the last element
      decoder_inputs = tf.pad(decoder_inputs,
                              [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    attention_bias = tf.cast(attention_bias, tf.float32)
    with tf.name_scope("add_pos_encoding"):
      length = tf.shape(decoder_inputs)[1]
      pos_encoding = TransformerUtils.get_position_encoding(
        length, self.args.hidden_size)
      pos_encoding = tf.cast(pos_encoding, tf.float32)
      decoder_inputs += pos_encoding
    if self.trainable:
      decoder_inputs = tf.nn.dropout(
        decoder_inputs, rate=self.args.dropout)

      # Run values
    decoder_self_attention_bias = TransformerUtils.get_decoder_self_attention_bias(
      length, dtype=tf.float32)
    outputs = self.decoder_stack(
      decoder_inputs,
      enc_output,
      decoder_self_attention_bias,
      attention_bias,
      training=self.trainable)

    logits = self.tgt_emb_layer(outputs, mode="linear")
    logits = tf.cast(logits, tf.float32)
    # increase the temperature of the logits before
    # turning them into probability distribution
    if self.args.distillation == 'True':
      logits = tf.math.divide(logits, self.temp)

    return logits

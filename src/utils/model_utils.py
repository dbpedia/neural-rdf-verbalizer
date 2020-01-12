""" Transformer model helper methods """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import re
import unicodedata

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

_NEG_INF = -1e9


def _set_up_dirs(args):
  if args.use_colab is None:
    EvalResultsFile = 'eval_results.txt'
    TestResults = 'test_results.txt'
    OUTPUT_DIR = 'ckpts/' + args.lang
    log_dir = 'data/logs'
    log_file = log_dir + args.lang + '_' + args.enc_type + '_' + str(args.emb_dim) + '.log'
    if not os.path.isdir(OUTPUT_DIR):
      os.makedirs(OUTPUT_DIR)
  else:
    from google.colab import drive
    drive.mount('/content/gdrive')
    OUTPUT_DIR = '/content/gdrive/My Drive/ckpts/' + args.lang
    EvalResultsFile = OUTPUT_DIR + '/eval_results.txt'
    TestResults = OUTPUT_DIR + '/test_results.txt'
    log_dir = OUTPUT_DIR + '/logs'
    log_file = log_dir + args.lang + '_' + args.enc_type + '_' + str(args.emb_dim) + '.txt'
    if not os.path.isdir(OUTPUT_DIR):
      os.makedirs(OUTPUT_DIR)

  return OUTPUT_DIR, EvalResultsFile, TestResults, log_file, log_dir


def _tensorize(vocab, text):
  """
  Function to convert texts into number sequences first, and then
  add padding. Basically, tensorising them.
  :param vocab: The vocab which is used to lookup ids
  :type vocab: tf.tokenizer obj
  :param text: A list of sentences or a text file
  :type text: list
  :return: tensorised text data
  :rtype: tf.tensor
  """
  tensor = vocab.texts_to_sequences(text)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor


def read_sentencepiece_vocab(filepath):
  voc = []
  with open(filepath, encoding='utf-8') as fi:
    for line in fi:
      voc.append(line.split("\t")[0])
      # skip the first <unk> token
  voc = voc[1:]
  return voc


def parse_sentencepiece_token(token):
  if token.startswith("▁"):
    return token[1:]
  else:
    return "##" + token


def max_length(tensor):
  return max(len(t) for t in tensor)


def convert(lang, tensor):
  for t in tensor:
    if t != 0:
      print("%d ----> %s" % (t, lang.index_word[t.numpy()]))


def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                 if unicodedata.category(c) != 'Mn')


def PreProcessSentence(w, sentencepiece, lang):
  """
  Preprocess a sentence by cleaning it, making everything
  lower case etc.
  If sentencepiece is being used then don't make spaces
  between punctuations and words, as it has it's own
  tokenizer.
  :param lang: Language of sentence
  :type lang: str
  :param w: Sentence to be preprocessed
  :type w: str
  :param sentencepiece: Is sentencepiece being used ?
  :type sentencepiece: str
  :return:Preprocessed sentence
  :rtype:str
  """
  w = unicode_to_ascii(w.lower().strip())
  if sentencepiece == 'False':
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    if lang == 'eng':
      w = re.sub(r"[^a-z0-9A-Z?.!,¿]+", " ", w)

  w = w.rstrip().strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w


def model_summary(model):
  """
  Gives summary of model and its params
  :param model: the model
  :type model: tf.keras.model object
  :return: summary text
  :rtype: write obj
  """
  model_vars = model.trainable_variables
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_position_encoding(length, hidden_size, min_timescale=1.0,
                          max_timescale=1.0e4):
  """
  Function to get the positional encoding for sequences to
  impart structural information

  :param length: sequence length
  :type length: int
  :param hidden_size: size of hidden state
  :type hidden_size: Tensor
  :param min_timescale:  Minimum scale that will be applied at each position
  :type min_timescale: float
  :param max_timescale: Maximum scale that will be applied at each position
  :type max_timescale: float
  :return: Tensor with shape [length, hidden_size]
  :rtype: tf.Tensor
  """
  position = tf.cast(tf.range(length), tf.float32)
  num_timescales = hidden_size // 2
  log_timescale_increment = (
          math.log(float(max_timescale) / float(min_timescale)) /
          (tf.cast(num_timescales, tf.float32) - 1))
  inv_timescales = min_timescale * tf.exp(
    tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

  return signal


def get_decoder_self_attention_bias(length):
  """
  Calculate bias for decoder that maintains model's autoregressive property.
  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.
  :param length: length of sequences
  :type length:int
  :return: float tensor of shape [1, 1, length, length]
  :rtype: tf.Tensor
  """
  with tf.name_scope("decoder-self_attention_bias"):
    valid_locs = tf.linalg.band_part(tf.ones([length, length]), -1, 0)
    valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
    decoder_bias = _NEG_INF * (1.0 - valid_locs)
  return decoder_bias


def get_padding(x, padding_values=0):
  """
  Returns float tensor representing the padding values
  :param x:  input tensor
  :type x: tf.Tensor
  :param padding_values: padding_value
  :type padding_values: int
  :return: tensor with same shape as x containing values 0 or 1.
  :rtype: tf.tensor
  """
  with tf.name_scope("padding"):
    return tf.cast(tf.equal(x, padding_values), tf.float32)


def get_padding_bias(x):
  """
  Calculate bias tensor from padding values in tensor
  :param x: input tensor
  :type x: tensor
  :return: attention bias
  :rtype: tensor [batch_size, 1, 1, length]
  """
  with tf.name_scope("attention_bias"):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF
    attention_bias = tf.expand_dims(
      tf.expand_dims(attention_bias, axis=1), axis=1)

    return attention_bias


def loss_function(real, pred, loss_object):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth_v)

  return output, attention_weights


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  sines = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  cosines = np.cos(angle_rads[:, 1::2])

  pos_encoding = np.concatenate([sines, cosines], axis=-1)

  pos_encoding = pos_encoding[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions so that we can add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
    q, k, v, None)
  print('Attention weights are:')
  print(temp_attn)
  print('Output is:')
  print(temp_out)


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
    tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
    tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask


def create_transgat_masks(tar):
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return combined_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def Padding(tensor, max_length):
  """
  Pads the given tensor to a maximum sequence length along
  axis 1.
  for ex -
  let the tensor be [1,2,3,4] if th given max_length is 5
  the tensor becomes [1,2,34,0]
  Mostly used to pad the target sentences of the multilingual
  model and the node_list of all models,

  :param tensor:A tf tensor
  :type tensor:tf.tensor
  :param max_length:Dimension along axis 1, of the new tensor
  :type max_length:int
  :return:The padded tensor
  :rtype:tf tensor.
  """

  padding = tf.constant([[0, 0], [0, max_length - tensor.shape[1]]])
  padded_tensor = tf.pad(tensor, padding, mode='CONSTANT')

  return padded_tensor

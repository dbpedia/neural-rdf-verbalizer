"""Script to load the target sentences and process, save them
as tf.data files
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pickle

import sentencepiece as spm
import tensorflow as tf

from src.utils.model_utils import max_length, _tensorize, Padding as padding


def LoadDataset(train_path, eval_path, test_path,
                vocab_path, sentencepiece):
  # load the train and eval datasets
  with open(train_path, 'rb') as f:
    train_set = pickle.load(f)
  with open(eval_path, 'rb') as f:
    eval_set = pickle.load(f)
  with open(test_path, 'rb') as f:
    test_set = pickle.load(f)

  train_inp, train_tgt = zip(*train_set)
  eval_inp, eval_tgt = zip(*eval_set)

  # load the vocab
  if sentencepiece == 'True':
    sp = spm.SentencePieceProcessor()
    sp.load(vocab_path)

    input_tensor = [sp.encode_as_ids(w) for w in train_inp]
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 padding='post')
    target_tensor = [sp.encode_as_ids(w) for w in train_tgt]
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  padding='post')
    eval_inp = [sp.encode_as_ids(w) for w in eval_inp]
    eval_inp = tf.keras.preprocessing.sequence.pad_sequences(eval_inp,
                                                             padding='post')
    eval_tgt = [sp.encode_as_ids(w) for w in eval_tgt]
    eval_tgt = tf.keras.preprocessing.sequence.pad_sequences(eval_tgt,
                                                             padding='post')

    test_inp = [sp.encode_as_ids(w) for w in test_set]
    test_inp = tf.keras.preprocessing.sequence.pad_sequences(test_inp,
                                                             padding='post')

    return input_tensor, target_tensor, \
           eval_inp, eval_tgt, test_inp, sp, max_length(target_tensor)
  else:
    with open(vocab_path, 'rb') as f:
      vocab = pickle.load(f)

    input_tensor = _tensorize(vocab, train_inp)
    target_tensor = _tensorize(vocab, train_tgt)
    eval_inp = _tensorize(vocab, eval_inp)
    eval_tgt = _tensorize(vocab, eval_tgt)
    test_inp = _tensorize(vocab, test_set)

    return input_tensor, target_tensor, \
           eval_inp, eval_tgt, test_inp, vocab, max_length(target_tensor)


def LoadGatDataset(train_path, eval_path, test_path, srv_vocab,
                   tgt_vocab, opt, sentencepiece, lang, num_examples=None):
  train_ = {}
  eval_ = {}
  test_ = {}
  # load the train and eval datasets
  with open(train_path, 'rb') as f:
    train_set = pickle.load(f)
  with open(eval_path, 'rb') as f:
    eval_set = pickle.load(f)
  with open(test_path, 'rb') as f:
    test_set = pickle.load(f)

  # load vocab
  if sentencepiece == 'True':
    sp = spm.SentencePieceProcessor()
    sp.load(tgt_vocab)
  with open(srv_vocab, 'rb') as f:
    src_vocab = pickle.load(f)

  train_input, train_tgt = zip(*train_set)
  eval_input, eval_tgt = zip(*eval_set)
  (train_nodes, train_labels, train_node1, train_node2) = zip(*train_input)
  (eval_nodes, eval_labels, eval_node1, eval_node2) = zip(*eval_input)
  (test_nodes, test_labels, test_node1, test_node2) = zip(*test_set)

  train_["train_node_tensor"] = _tensorize(src_vocab, train_nodes)
  train_["train_label_tensor"] = _tensorize(src_vocab, train_labels)
  train_["train_node1_tensor"] = _tensorize(src_vocab, train_node1)
  train_["train_node2_tensor"] = _tensorize(src_vocab, train_node2)

  eval_["eval_node_tensor"] = _tensorize(src_vocab, eval_nodes)
  eval_["eval_label_tensor"] = _tensorize(src_vocab, eval_labels)
  eval_["eval_node1_tensor"] = _tensorize(src_vocab, eval_node1)
  eval_["eval_node2_tensor"] = _tensorize(src_vocab, eval_node2)

  test_["test_node_tensor"] = _tensorize(src_vocab, test_nodes)
  test_["test_label_tensor"] = _tensorize(src_vocab, test_labels)
  test_["test_node1_tensor"] = _tensorize(src_vocab, test_node1)
  test_["test_node2_tensor"] = _tensorize(src_vocab, test_node2)

  if sentencepiece == 'True':
    train_tgt_tensor = [sp.encode_as_ids(w) for w in train_tgt]
    train_["train_tgt_tensor"] = tf.keras.preprocessing.sequence.pad_sequences(train_tgt_tensor, padding='post')
    eval_tgt_tensor = [sp.encode_as_ids(w) for w in eval_tgt]
    eval_["eval_tgt_tensor"] = tf.keras.preprocessing.sequence.pad_sequences(eval_tgt_tensor, padding='post')
    target_vocab = sp
  else:
    train_tgt_tensor = src_vocab.texts_to_sequences(train_tgt)
    train_["train_tgt_tensor"] = tf.keras.preprocessing.sequence.pad_sequences(train_tgt_tensor, padding='post')
    eval_tgt_tensor = src_vocab.texts_to_sequences(eval_tgt)
    eval_["eval_tgt_tensor"] = tf.keras.preprocessing.sequence.pad_sequences(eval_tgt_tensor, padding='post')
    target_vocab = src_vocab

  return (train_, eval_, test_, src_vocab, target_vocab, max_length(train_tgt_tensor))


def GetDataset(args):
  input_tensor, target_tensor, \
  eval_tensor, eval_tgt, test_inp, lang, max_seq_len = LoadDataset(args.train_path, args.eval_path, args.test_path,
                                                                   args.src_vocab, args.sentencepiece)

  BUFFER_SIZE = len(input_tensor)
  BATCH_SIZE = args.batch_size
  steps_per_epoch = len(input_tensor) // BATCH_SIZE

  if args.sentencepiece == 'True':
    vocab_size = lang.get_piece_size()
  else:
    vocab_size = len(lang.word_index) + 1

  dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
  eval_set = tf.data.Dataset.from_tensor_slices((eval_tensor, eval_tgt))
  test_set = tf.data.Dataset.from_tensor_slices((test_inp))
  dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
  eval_set = eval_set.batch(BATCH_SIZE, drop_remainder=True)
  test_set = test_set.batch(BATCH_SIZE, drop_remainder=True)
  dataset_size = target_tensor.shape[0]

  return (dataset, eval_set, test_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
          vocab_size, lang, dataset_size, max_seq_len)


def GetGATDataset(args, set=None):
  (train, eval, test, src_vocab, tgt_vocab, max_length_targ) = LoadGatDataset(args.train_path,
                                                                              args.eval_path,
                                                                              args.test_path, args.src_vocab,
                                                                              args.tgt_vocab, args.opt,
                                                                              args.sentencepiece, args.lang)

  node_tensor = padding(train["train_node_tensor"], 16)
  label_tensor = padding(train["train_label_tensor"], 16)
  node1_tensor = padding(train["train_node1_tensor"], 16)
  node2_tensor = padding(train["train_node2_tensor"], 16)

  eval_nodes = padding(eval["eval_node_tensor"], 16)
  eval_labels = padding(eval["eval_label_tensor"], 16)
  eval_node1 = padding(eval["eval_node1_tensor"], 16)
  eval_node2 = padding(eval["eval_node2_tensor"], 16)

  test_nodes = padding(test["test_node_tensor"], 16)
  test_labels = padding(test["test_label_tensor"], 16)
  test_node1 = padding(test["test_node1_tensor"], 16)
  test_node2 = padding(test["test_node2_tensor"], 16)

  print('\nTrain Tensor shapes (nodes, labels, node1, node2, target) : ')
  print(node_tensor.shape, label_tensor.shape, node1_tensor.shape, node2_tensor.shape,
        train["train_tgt_tensor"].shape)
  print('\nEval Tensor shapes (nodes, labes, node1, node2) : ')
  print(eval_nodes.shape, eval_labels.shape, eval_node1.shape, eval_node2.shape, eval["eval_tgt_tensor"].shape)
  print('\nTest Tensor shapes (nodes, labes, node1, node2) : ')
  print(test_nodes.shape, test_labels.shape, test_node1.shape, test_node2.shape)

  TRAIN_BUFFER_SIZE = len(train["train_tgt_tensor"])
  EVAL_BUFFER_SIZE = len(eval["eval_tgt_tensor"])
  BATCH_SIZE = args.batch_size
  steps_per_epoch = len(train["train_tgt_tensor"]) // BATCH_SIZE
  src_vocab_size = len(src_vocab.word_index) + 1
  if args.sentencepiece == 'True':
    tgt_vocab_size = tgt_vocab.get_piece_size()
  else:
    tgt_vocab_size = len(tgt_vocab.word_index) + 1

  dataset_size = train["train_tgt_tensor"].shape[0]

  dataset = tf.data.Dataset.from_tensor_slices((node_tensor, label_tensor,
                                                node1_tensor, node2_tensor, train["train_tgt_tensor"])).shuffle(
    TRAIN_BUFFER_SIZE)
  dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

  eval_set = tf.data.Dataset.from_tensor_slices((eval_nodes, eval_labels,
                                                 eval_node1, eval_node2, eval["eval_tgt_tensor"])).shuffle(
    EVAL_BUFFER_SIZE)
  eval_set = eval_set.batch(BATCH_SIZE, drop_remainder=True)

  test_set = tf.data.Dataset.from_tensor_slices((test_nodes, test_labels,
                                                 test_node1, test_node2))
  test_set = test_set.batch(BATCH_SIZE, drop_remainder=True)

  if args.debug_mode == "True":
    dataset = dataset.take(1)

  if set == None:
    return (dataset, eval_set, test_set, TRAIN_BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
            src_vocab_size, src_vocab, tgt_vocab_size, tgt_vocab,
            max_length_targ, dataset_size)
  elif set == 'test':
    return (test_set, TRAIN_BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
            src_vocab_size, src_vocab, tgt_vocab_size, tgt_vocab)

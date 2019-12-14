"""
To translate the entirety of test set and calculate bleu score.
"""

import argparse
import os
import pickle

import sentencepiece as spm
import tensorflow as tf

from src.models import GraphAttentionModel
from src.utils.PreprocessingUtils import PreProcess
from src.utils.model_utils import Padding as padding

parser = argparse.ArgumentParser(description="Main Arguments")

parser.add_argument(
  '--model', type=str, required=True, help='The model used to verbalise the triple ')
parser.add_argument(
  '--lang', type=str, required=True, help='Language of the target sentence ')
parser.add_argument(
  '--triples', type=str, required=True, help='Path to the triple file ')
parser.add_argument(
  '--sentencepiece', type=str, required=True, help='Use sentencepiece or not ')
parser.add_argument(
  '--batch_size', type=str, required=True, help='Batch size to do inference ')

args = parser.parse_args()


def LoadModel(model, lang):
  """
  Function to load the model from stored checkpoint.
  :param args: All arguments that were given to train file
  :type args: Argparse object
  :return: model
  :rtype: tf.keras.Model
  """

  if model == 'gat':

    log_dir = 'data/logs'
    with open(log_dir + '/' + lang + '_model_params', 'rb') as fp:
      params = pickle.load(fp)

    model_args = params['args']

    if model_args.use_colab is None:
      OUTPUT_DIR = 'ckpts/' + model_args.lang
      if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
    else:
      from google.colab import drive

      drive.mount('/content/gdrive')
      OUTPUT_DIR = '/content/gdrive/My Drive/ckpts/' + model_args.lang
      if not os.path.isdir(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

    if model_args.enc_type == 'gat' and model_args.dec_type == 'transformer':
      OUTPUT_DIR += '/' + model_args.enc_type + '_' + model_args.dec_type

      # Load the vocabs
      with open('vocabs/' + model_args.model + '/' +
                lang + '/' + model_args.opt + '_src_vocab', 'rb') as fp:
        src_vocab = pickle.load(fp)
      # loading the target vocab
      model_args.sentencepiece = 'False'
      if model_args.sentencepiece == 'True':
        sp = spm.SentencePieceProcessor()
        sp.load('vocabs/' + model_args.model + '/' +
                lang + '/' + 'train_tgt.model')
        tgt_vocab = sp
      else:
        tgt_vocab = src_vocab

      print('Loaded ' + lang + ' Parameters..')
      model = GraphAttentionModel.TransGAT(params['args'], params['src_vocab_size'], src_vocab,
                                           params['tgt_vocab_size'], params['max_tgt_length'], tgt_vocab)

      # Load the latest checkpoints
      optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.98,
                                         epsilon=1e-9)

      ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer
      )

      ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
      if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    print('Loaded ' + lang + ' model !')

    return model, src_vocab, tgt_vocab


def _tensorize_triples(nodes, labels,
                       node1, node2, src_vocab):
  node_tensor = src_vocab.texts_to_sequences(nodes)
  label_tensor = src_vocab.texts_to_sequences(labels)
  node1_tensor = src_vocab.texts_to_sequences(node1)
  node2_tensor = src_vocab.texts_to_sequences(node2)
  node_tensor = padding(
    tf.keras.preprocessing.sequence.pad_sequences(node_tensor, padding='post'), 16)
  label_tensor = padding(
    tf.keras.preprocessing.sequence.pad_sequences(label_tensor, padding='post'), 16)
  node1_tensor = padding(
    tf.keras.preprocessing.sequence.pad_sequences(node1_tensor, padding='post'), 16)
  node2_tensor = padding(
    tf.keras.preprocessing.sequence.pad_sequences(node2_tensor, padding='post'), 16)

  dataset = tf.data.Dataset.from_tensor_slices((node_tensor, label_tensor,
                                                node1_tensor, node2_tensor))
  dataset = dataset.batch(int(args.batch_size), drop_remainder=False)

  return dataset


if __name__ == "__main__":
  model, src_vocab, tgt_vocab = LoadModel(args.model, args.lang)
  nodes, labels, node1, node2 = PreProcess(args.triples, args.lang)

  dataset = _tensorize_triples(nodes, labels, node1, node2, src_vocab)
  results = []

  for (batch, (nodes, labels, node1, node2)) in (enumerate(dataset)):
    predictions = model(nodes, labels, node1,
                        node2, targ=None, mask=None)
    pred = [(predictions['outputs'].numpy().tolist())]
    if args.sentencepiece == 'True':
      for i in range(len(pred[0])):
        sentence = (tgt_vocab.DecodeIds(list(pred[0][i])))
        sentence = sentence.partition("<start>")[2].partition("<end>")[0]
        results.append(sentence)
    else:
      for i in pred:
        sentences = tgt_vocab.sequences_to_texts(i)
        sentence = [j.partition("<start>")[2].partition("<end>")[0] for j in sentences]
        for w in sentences:
          results.append(w)

  print(results)
  results_file = open('results.txt', 'w+')
  results_file.writelines(results)
  results_file.close()

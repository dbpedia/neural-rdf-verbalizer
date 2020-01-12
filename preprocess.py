""" Script to preprocess the .triple files to
create graphs using the networkx library, and
save the adjacency matrix as numpy array

 Takes in the .triple file which has each RDF triple
 from the dataset in <subject | predicate | object>
 form. Also uses networkx library to represent each
 example instance s a graph and get the adjacency matrix
 as a numpy array or a tf.Tensor

 Then creates a dataset file with each entry being the
 adjacency matrix of that graph, combined with nodes of
 the graph whose embeddings are used as node features
 are used as inputs to the Graph Neural Network encoder.
"""
import argparse
import io
import os
import pickle

import sentencepiece as spm
import tensorflow as tf
from loguru import logger

from src.utils.PreprocessingUtils import PreProcess
from src.utils.model_utils import PreProcessSentence

parser = argparse.ArgumentParser(description="preprocessor parser")
parser.add_argument(
  '--train_src', type=str, required=False, help='Path to train source file')
parser.add_argument(
  '--train_tgt', type=str, required=False, help='Path to train target file ')
parser.add_argument(
  '--eval_src', type=str, required=False, help='Path to eval source file')
parser.add_argument(
  '--eval_tgt', type=str, required=False, help='Path to eval target file')
parser.add_argument(
  '--spl_sym', type=str, required=False, help='Path to Special Symbols file')
parser.add_argument(
  '--test_src', type=str, required=False, help='Path to test source file')
parser.add_argument(
  '--model', type=str, required=True, help='Preprocess for GAT model or seq2seq model')
parser.add_argument(
  '--use_colab', type=bool, required=False, help='Use colab or not')
parser.add_argument(
  '--lang', type=str, required=True, help='Language of the dataset')
parser.add_argument(
  '--vocab_size', type=int, required=False, help='Size of target vocabulary')
parser.add_argument(
  '--max_seq_len', type=int, required=False, help='Maximum length of the sequence')
parser.add_argument(
  '--sentencepiece_model', type=str, required=False, help='SentencePiece model')
parser.add_argument(
  '--sentencepiece', type=str, required=True, help='Use SentencePiece or not ')

args = parser.parse_args()

if __name__ == '__main__':
  os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)

  if args.model == 'gat':
    train_nodes, train_labels, train_node1, train_node2 = PreProcess(args.train_src, args.lang)
    eval_nodes, eval_labels, eval_node1, eval_node2 = PreProcess(args.eval_src, args.lang)
    test_nodes, test_labels, test_node1, test_node2 = PreProcess(args.test_src, args.lang)

    # Build and save the vocab
    print('Building the  Source Vocab file... ')
    train_tgt = io.open(args.train_tgt, encoding='UTF-8').read().strip().split('\n')
    train_tgt = [PreProcessSentence(w, args.sentencepiece, args.lang) for w in train_tgt]
    # vocab_train_tgt = [tokenizer(w) for w in train_tgt]
    eval_tgt = io.open(args.eval_tgt, encoding='UTF-8').read().strip().split('\n')
    eval_tgt = [PreProcessSentence(w, args.sentencepiece, args.lang) for w in eval_tgt]

    vocab = tf.keras.preprocessing.text.Tokenizer(filters='')
    vocab.fit_on_texts(train_nodes)
    vocab.fit_on_texts(train_labels)
    vocab.fit_on_texts(train_node1)
    vocab.fit_on_texts(train_node2)
    vocab.fit_on_texts(eval_nodes)
    vocab.fit_on_texts(eval_labels)
    vocab.fit_on_texts(eval_node1)
    vocab.fit_on_texts(eval_node2)

    if args.sentencepiece == 'True':
      spm.SentencePieceTrainer.Train('--input={},{} --model_prefix=vocabs/{}/{}/train_vocab'
                                     ' --vocab_size={} --character_coverage=1.0 --model_type={}'.format(
        args.train_tgt, args.eval_tgt, args.model, args.lang,
        str(args.vocab_size), args.sentencepiece_model))
      sp = spm.SentencePieceProcessor()
      sp.load('vocabs/{}/{}/train_vocab.model'.format(args.model, args.lang))
      logger.info('Sentencepiece vocab size {}'.format(sp.get_piece_size()))
      target_vocab = sp
    else:
      vocab.fit_on_texts(train_tgt)
      vocab.fit_on_texts(eval_tgt)

    logger.info('Vocab Size : {}\n'.format(len(vocab.word_index)))

    train_input = list(zip(train_nodes, train_labels, train_node1, train_node2))
    eval_input = list(zip(eval_nodes, eval_labels, eval_node1, eval_node2))
    test_input = list(zip(test_nodes, test_labels, test_node1, test_node2))
    train_set = list(zip(train_input, train_tgt))
    eval_set = list(zip(eval_input, eval_tgt))
    logger.info('Train and eval dataset size : {} {} '.format(len(train_set), len(eval_set)))

    if args.use_colab is not None:
      from google.colab import drive

      drive.mount('/content/gdrive', force_remount=True)
      OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/{}/{}'.format(args.lang,
                                                                                 args.model)
      if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
      # save the vocab file
      os.makedirs(('vocabs/gat/{}'.format(args.lang)), exist_ok=True)
      with open(('vocabs/gat/{}/src_vocab'.format(args.lang)), 'wb+') as fp:
        pickle.dump(vocab, fp)

    else:
      OUTPUT_DIR = 'data/processed_graphs/{}/{}'.format(args.lang,
                                                        args.model)
      if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
      # save the vocab file
      os.makedirs(('vocabs/gat/{}'.format(args.lang)), exist_ok=True)
      with open(('vocabs/gat/{}/src_vocab'.format(args.lang)), 'wb+') as fp:
        pickle.dump(vocab, fp)

    print('Vocab file saved !\n')
    print('Preparing the Graph Network datasets...')

    with open(OUTPUT_DIR + '/' + '_train', 'wb') as fp:
      pickle.dump(train_set, fp)
    with open(OUTPUT_DIR + '/' + '_eval', 'wb') as fp:
      pickle.dump(eval_set, fp)
    with open(OUTPUT_DIR + '/' + '_test', 'wb') as fp:
      pickle.dump(test_input, fp)
    print('Dumped the train, eval and test datasets.')

  else:
    # Train the vocabs
    os.makedirs(('vocabs/{}/{}'.format(args.model, args.lang)), exist_ok=True)  # ready the directories

    print('Building the dataset...')

    train_src = io.open(args.train_src, encoding='UTF-8').read().strip().split('\n')
    train_src = [PreProcessSentence(w, args.sentencepiece, args.lang) for w in train_src]
    eval_src = io.open(args.eval_src, encoding='UTF-8').read().strip().split('\n')
    eval_src = [PreProcessSentence(w, args.sentencepiece, args.lang) for w in eval_src]
    train_tgt = io.open(args.train_tgt, encoding='UTF-8').read().strip().split('\n')
    train_tgt = [PreProcessSentence(w, args.sentencepiece, args.lang) for w in train_tgt]
    eval_tgt = io.open(args.eval_tgt, encoding='UTF-8').read().strip().split('\n')
    eval_tgt = [PreProcessSentence(w, args.sentencepiece, args.lang) for w in eval_tgt]
    test_src = io.open(args.test_src, encoding='UTF-8').read().strip().split('\n')
    test_src = [PreProcessSentence(w, args.sentencepiece, args.lang) for w in test_src]

    if args.sentencepiece == 'True':
      spm.SentencePieceTrainer.Train('--input={},{},{},{} --model_prefix=vocabs/{}/{}/train_vocab'
                                     ' --vocab_size={} --character_coverage=1.0 --model_type={}'.format(
        args.train_tgt, args.eval_tgt, args.train_src, args.eval_src,
        args.model, args.lang, str(args.vocab_size), args.sentencepiece_model
      ))

      sp = spm.SentencePieceProcessor()
      sp.load('vocabs/{}/{}/train_vocab.model'.format(args.model, args.lang))
      logger.info('Sentencepiece vocab size {}'.format(sp.get_piece_size()))
    else:
      vocab = tf.keras.preprocessing.text.Tokenizer(filters='')
      vocab.fit_on_texts(train_src)
      vocab.fit_on_texts(train_tgt)
      vocab.fit_on_texts(eval_src)
      vocab.fit_on_texts(eval_tgt)

      if args.use_colab is not None:
        from google.colab import drive

        drive.mount('/content/gdrive', force_remount=True)
        OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/{}/{}'.format(args.lang,
                                                                                   args.model)
        if not os.path.isdir(OUTPUT_DIR):
          os.makedirs(OUTPUT_DIR)
        # save the vocab file
        os.makedirs(('vocabs/{}/{}'.format(args.model, args.lang)), exist_ok=True)
        with open(('vocabs/{}/{}/vocab'.format(args.model, args.lang)), 'wb+') as fp:
          pickle.dump(vocab, fp)

      else:
        OUTPUT_DIR = 'data/processed_graphs/{}/{}'.format(args.lang, args.model)
        if not os.path.isdir(OUTPUT_DIR):
          os.makedirs(OUTPUT_DIR)
        # save the vocab file
        os.makedirs(('vocabs/{}/{}'.format(args.model, args.lang)), exist_ok=True)
        with open(('vocabs/{}/{}/vocab'.format(args.model, args.lang)), 'wb+') as fp:
          pickle.dump(vocab, fp)

      print('Vocab file saved !\n')

    train_set = zip(train_src, train_tgt)
    eval_set = zip(eval_src, eval_tgt)

    if args.use_colab is not None:
      from google.colab import drive

      drive.mount('/content/gdrive', force_remount=True)
      OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/{}/{}'.format(args.lang,
                                                                                 args.model)
      if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    else:
      OUTPUT_DIR = 'data/processed_graphs/{}/{}'.format(args.lang, args.model)
      if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    with open(OUTPUT_DIR + '/' + 'train', 'wb') as fp:
      pickle.dump(train_set, fp)
    with open(OUTPUT_DIR + '/' + 'eval', 'wb') as fp:
      pickle.dump(eval_set, fp)
    with open(OUTPUT_DIR + '/' + 'test', 'wb') as fp:
      pickle.dump(test_src, fp)
    print('Dumped the train and eval datasets.')

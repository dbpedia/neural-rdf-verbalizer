from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import pickle
import argparse
from pathlib import Path
import numpy as np
import os
#from src.data_loader import max_length, convert, get_gat_dataset

languages = ['eng', 'rus', 'ger']
parser = argparse.ArgumentParser(description="Main Arguments")

# model paramteres
parser.add_argument(
    '--model', default='rnn', type=str, required=True,
    help='Type of encoder Transformer | gat | rnn')
parser.add_argument(
    '--opt', default='rnn', type=str, required=True,
    help='Type of decoder Transformer | rnn')
parser.add_argument(
    '--lang', default='rnn', type=str, required=True,
    help='Type of decoder Transformer | rnn')
parser.add_argument(
    '--use_colab', default=None, type=str, required=False,
    help='Type of decoder Transformer | rnn')

def load_gat_multidataset(args):
    """

    :param args:
    :type args:
    :return:
    :rtype:
    """
    dataset = {}
    CUR_DIR = os.getcwd()
    levels_up = 1
    DATA_PATH = (os.path.normpath(os.path.join(*([CUR_DIR]+[".."]*levels_up))))+'/data/processed_graphs/'

    TRAIN_DIRS = [DATA_PATH+lang+'/'+args.model+'/'+args.opt+'_train' for lang in languages]
    EVAL_DIRS = [DATA_PATH + lang + '/' + args.model + '/' + args.opt + '_eval' for lang in languages]

    vocab = tf.keras.preprocessing.text.Tokenizer(filters='')

    for (train_dir, eval_dir, lang) in zip(TRAIN_DIRS, EVAL_DIRS, languages):
        with open(train_dir, 'rb') as f:
            train_set = pickle.load(f)
        with open(eval_dir, 'rb') as f:
            eval_set = pickle.load(f)

        train_input, train_tgt = zip(*train_set)
        eval_input, eval_tgt = zip(*eval_set)

        (dataset[lang+'_train_nodes'], dataset[lang+'_train_labels'],
         dataset[lang+'_train_node1'], dataset[lang+'_train_node2']) = zip(*train_input)
        (dataset[lang + '_eval_nodes'], dataset[lang + '_eval_labels'],
         dataset[lang + '_eval_node1'], dataset[lang + '_eval_node2']) = zip(*eval_input)
        dataset[lang + '_train_tgt'] = train_tgt
        dataset[lang + '_eval_tgt'] = eval_tgt

        [x.append(lang) for x in dataset[lang+'_train_nodes']]
        #fit the vocabs on the dataset
        vocab.fit_on_texts(dataset[lang+'_train_nodes'])
        vocab.fit_on_texts(dataset[lang + '_train_labels'])
        vocab.fit_on_texts(dataset[lang + '_train_node1'])
        vocab.fit_on_texts(dataset[lang + '_train_node2'])
        vocab.fit_on_texts(dataset[lang + '_train_tgt'])

    #save the dataset and vocab
    os.makedirs(('vocabs/gat/' + args.lang), exist_ok=True)
    with open(('vocabs/gat/' + args.lang + '/' + args.opt + '_vocab'), 'wb+') as fp:
        pickle.dump(vocab, fp)

    if args.use_colab is not None:
        from google.colab import drive

        drive.mount('/content/gdrive', force_remount=True)
        OUTPUT_DIR = '/content/gdrive/My Drive/data/processed_graphs/' + args.lang + '/' + args.model
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

    else:
        OUTPUT_DIR = 'data/processed_graphs/' + args.lang + '/' + args.model
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
    with open(OUTPUT_DIR + '/' + args.opt + '_train', 'wb') as fp:
        pickle.dump(dataset, fp)
    print('Dumped the dataset and Vocab !')

    return dataset, vocab

def process_gat_multidataset(dataset, vocab):
    for lang in languages:
        dataset[lang+'_train_nodes'] = vocab.texts_to_sequences(dataset[lang+'_train_nodes'])
        dataset[lang + '_train_labels'] = vocab.texts_to_sequences(dataset[lang + '_train_labels'])
        dataset[lang + '_train_node1'] = vocab.texts_to_sequences(dataset[lang + '_train_node1'])
        dataset[lang + '_train_node2'] = vocab.texts_to_sequences(dataset[lang + '_train_node2'])

        dataset[lang + '_train_tgt'] = vocab.texts_to_sequences(dataset[lang + '_train_tgt'])

        dataset[lang + '_train_nodes'] = tf.keras.preprocessing.sequence.pad_sequences(dataset[lang+'_train_nodes'],
                                                                                        padding='post')
        dataset[lang + '_train_labels'] = tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_train_labels'],
                                                                                       padding='post')
        dataset[lang + '_train_node1'] = tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_train_node1'],
                                                                                       padding='post')
        dataset[lang + '_train_node2'] = tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_train_node2'],
                                                                                       padding='post')
        dataset[lang + '_train_tgt'] = tf.keras.preprocessing.sequence.pad_sequences(dataset[lang + '_train_tgt'],
                                                                                       padding='post')




if __name__ == "__main__":
    args = parser.parse_args()
    load_gat_multidataset(args)